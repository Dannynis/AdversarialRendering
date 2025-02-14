

import kaolin
import kaolin as kal
import pickle
import torch
import objaverse


import math
import matplotlib.pyplot as plt

from kaolin.render.lighting import SgLightingParameters
import numpy as np

from datetime import datetime

IMAGE_SIZE = 1024

def make_camera(eye):
  return kal.render.camera.Camera.from_args(eye=torch.tensor(eye),
                                         at=torch.tensor([0., 0., 0.]),
                                         up=torch.tensor([0., 1., 0]),
                                         fov=math.pi * 45 / 180,
                                            near=0.1, far=10000.,
                                         width=IMAGE_SIZE,
                                            height=IMAGE_SIZE,
                                            device='cuda')
forbidden_theta = []
forbidden_phi = []

resolution = 20
for theta in np.linspace(0.0001, 0.6, 3):
    forbidden_theta.append(theta)

for phi in np.linspace(0.0001, 10, resolution):
    forbidden_phi.append(phi)

forbidden_theta = set(forbidden_theta)
forbidden_phi = set(forbidden_phi)

def random_polar(r_range, phi_range, theta_range):
  done = False
  while not done:
    r = np.random.uniform(r_range[0], r_range[1])
    theta = np.random.uniform(theta_range[0], theta_range[1])
    phi = np.random.uniform(phi_range[0], phi_range[1])
    if theta not in forbidden_theta and phi not in forbidden_phi:
      done = True
  return [r, theta, phi]


def polar_to_cartesian(r, phi, theta):
  y = r * math.cos(theta)
  z = r * math.sin(theta) * math.cos(phi)
  x = r * math.sin(theta) * math.sin(phi)
  return [x,y,z]

def random_light(strength_range = [8,15],suns_range=[1, 5], phi_range=[0, math.pi * 2], theta_range=[0, math.pi / 2]):
  n_suns = int(np.random.uniform(suns_range[0],suns_range[1]))
  light_directions = []
  for i in range(n_suns):
    [r, theta, phi] = random_polar(r_range=[1, 5], phi_range=phi_range, theta_range=theta_range)
    direction = np.array(polar_to_cartesian(r, phi, theta))
    direction = direction / np.sqrt(np.sum(direction * direction))
    light_directions.append(direction)

  light_directions = torch.tensor(np.array(light_directions)).cuda()

  strength = np.random.uniform(strength_range[0],strength_range[1])
  lighting = SgLightingParameters.from_sun(light_directions.float(), strength).cuda()

  return lighting, (strength,light_directions)

def polar_camera_and_light(r, phi, theta):
  eye = polar_to_cartesian(r, phi, theta)
  camera = make_camera(eye)
  eye = np.array(eye)
  eye_norm = np.sqrt(np.sum(eye * eye))

  n_suns = int(np.random.uniform(1, 5))
  light_directions = []
  light_direction = torch.tensor(eye / eye_norm, dtype=torch.float32).view(1, 1, 3).cuda()
  strength = np.random.uniform(4, 10)
  lighting = SgLightingParameters.from_sun(light_direction.float(), strength).cuda()
  return camera, lighting

theta_eps = 0.3

def random_camera_and_light(r_range = [0, 5], phi_range=[0, math.pi * 2], theta_range=[ math.pi / 2 - theta_eps,0]):
  [r, theta, phi] = random_polar(r_range, phi_range, theta_range)
  return polar_camera_and_light(r, phi, theta), (r, phi, theta)


def render(in_cam, mesh, lighting, pbr_mat=None):
    if pbr_mat is not None:
      render_res = kal.render.easy_render.render_mesh(in_cam, mesh, lighting=lighting, custom_materials = [pbr_mat])
    else:
      render_res = kal.render.easy_render.render_mesh(in_cam, mesh, lighting=lighting)
    img = render_res[kal.render.easy_render.RenderPass.render].squeeze(0).clamp(0, 1)
    return img

from datasets import  VerificationMode, load_dataset
import random 
ds = load_dataset("barkermrl/imagenet-a", num_proc=3)

images = []
n = 0
for x in ds["train"]:
  img = x["image"]
  l = min(img.width, img.height)
  img = img.crop((0, 0, l, l))
  img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
  images.append(img)
  n += 1
  if n > 500:
    break

def random_image():
  return random.choice(images)

import torchvision
import random 

def render_depth(mesh, cam, depth_slack=0.2):
  vertices_camera = cam.extrinsics.transform(mesh.vertices)
  vertices_image_full = cam.intrinsics.transform(vertices_camera)
  vertices_image = vertices_image_full[:, :, :2]
  vertices_image = vertices_image_full[:, :, :2]

  face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, mesh.faces)
  face_vertices_image_full = kal.ops.mesh.index_vertices_by_faces(vertices_image_full, mesh.faces)
  face_vertices_image = face_vertices_image_full[:, :, :, :2]
  face_vertices_image_z = face_vertices_image_full[:, :, :, 2:]
  face_normals = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)


  face_attributes = [face_vertices_image_z]
  image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
      cam.intrinsics.height, cam.intrinsics.width, face_vertices_camera[:, :, :, -1],
      face_vertices_image, face_attributes, face_normals[:, :, -1],
      rast_backend='cuda')

  depth = image_features[0].squeeze()

  object_mask = (depth > 0.0)

  depth_max = torch.max(depth[object_mask])
  depth_min = torch.min(depth[object_mask])
  depth = torch.where(object_mask, 1.0 - (depth - depth_min) / (depth_max - depth_min), 0.0)

  # slack avoids the back of the object being at depth 0 ; this is not good for controlnet
  depth = torch.where(object_mask, (depth + depth_slack) / (1.0 + depth_slack), 0.0)
  return depth.squeeze(), object_mask.squeeze()

def add_background(image, mask, background):
  mask = mask.unsqueeze(-1).repeat(1, 1, 3)
  background = torchvision.transforms.functional.pil_to_tensor(background).cuda().permute((1, 2, 0)) / 255.0
  image[~mask] = background[~mask]
  return image

def render2(in_cam, mesh, lighting,
           max_mesh_noise_intensity=0.05,
           max_render_noise_intensity=0.05,
           random_bg_prob=1, augmentation_func=None, background=None):
  # mesh = add_random_material_noise(mesh, max_intensity=max_mesh_noise_intensity)
  render_res = kal.render.easy_render.render_mesh(in_cam, mesh, lighting=lighting)
  image = render_res[kal.render.easy_render.RenderPass.render].squeeze(0).clamp(0, 1)
  # render_noise = random_noise(IMAGE_SIZE, 3, max_iterations=10, max_intensity=0.1).permute((1, 2, 0)).abs()
  # image = torch.clamp(image + render_noise, 0, 1)

  # convert normals to standard normal map colors
  normals = render_res["normals"].squeeze()
  mask = (normals == 0).all(dim=-1, keepdim=True)
  mask = torch.cat([torch.zeros((mask.shape[0], mask.shape[1], 2), device="cuda", dtype=torch.bool), mask], dim=-1)
  normals[mask] = 1.0
  normals = normals / torch.norm(normals, dim=-1, keepdim=True)
  normals = (1 + normals) / 2

  # additional depth render as there is no z-buffer pass in easy_render
  depth, mask = render_depth(mesh, in_cam)

  if augmentation_func:
    image = augmentation_func(image)

  if random.random() < random_bg_prob:
    if background is None:
      background = random_image()
    image = add_background(image, mask, background)

  return image, depth, normals, mask

def render(in_cam, mesh, lighting,
           max_mesh_noise_intensity=0.05,
           max_render_noise_intensity=0.05,
           random_bg_prob=1, augmentation_func=None ,background=None):
  
  image, depth, normals, mask = render2(in_cam, mesh, lighting, max_mesh_noise_intensity, max_render_noise_intensity, random_bg_prob, augmentation_func, background)
  return image

def render_with_mask(in_cam, mesh, lighting,
           max_mesh_noise_intensity=0.05,
           max_render_noise_intensity=0.05,
           random_bg_prob=1, augmentation_func=None ,background=None):
  
  image, depth, normals, mask = render2(in_cam, mesh, lighting, max_mesh_noise_intensity, max_render_noise_intensity, random_bg_prob, augmentation_func, background)
  return image, mask
