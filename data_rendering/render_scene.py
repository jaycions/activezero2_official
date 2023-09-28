import os
import os.path as osp
import sys

_ROOT_DIR = os.path.abspath(osp.join(osp.dirname(__file__), ".."))
sys.path.insert(0, _ROOT_DIR)
from loguru import logger
from PIL import Image

from data_rendering.utils.render_utils import *


def render_scene(
    sim: sapien.Engine,
    renderer: sapien.KuafuRenderer,
    scene_id,
    repo_root,
    target_root,
    spp,
    num_views,
    rand_pattern,
    fixed_angle,
    primitives,
    primitives_v2,
    rand_lighting,
    rand_table,
    rand_env,
):
    materials_root = os.path.join(repo_root, "data_rendering/materials")

    scene_config = sapien.SceneConfig()
    scene_config.solver_iterations = 25
    scene_config.solver_velocity_iterations = 2
    scene_config.enable_pcm = False
    scene_config.default_restitution = 0
    scene_config.default_dynamic_friction = 0.5
    scene_config.default_static_friction = 0.5
    scene = sim.create_scene(scene_config)
    scene.set_timestep(1 / 240)

    if not rand_env:
        ground_material = renderer.create_material()
        ground_material.base_color = np.array([10, 10, 10, 256]) / 256
        ground_material.specular = 0.5
        scene.add_ground(-2.0, render_material=ground_material)

    table_pose_np = np.loadtxt(os.path.join(repo_root, "data_rendering/materials/optical_table/pose.txt"))
    table_pose = sapien.Pose(table_pose_np[:3], table_pose_np[3:])

    if rand_table:
        load_rand_table(scene, os.path.join(repo_root, "data_rendering/materials/optical_table"), renderer, table_pose)
    else:
        load_table(scene, os.path.join(repo_root, "data_rendering/materials/optical_table"), renderer, table_pose)

    # Add camera
    cam_intrinsic_base = np.loadtxt(os.path.join(materials_root, "cam_intrinsic_base.txt"))
    cam_ir_intrinsic_base = np.loadtxt(os.path.join(materials_root, "cam_ir_intrinsic_base.txt"))
    cam_intrinsic_hand = np.loadtxt(os.path.join(materials_root, "cam_intrinsic_hand.txt"))
    cam_ir_intrinsic_hand = np.loadtxt(os.path.join(materials_root, "cam_ir_intrinsic_hand.txt"))
    cam_irL_rel_extrinsic_base = np.loadtxt(
        os.path.join(materials_root, "cam_irL_rel_extrinsic_base.txt")
    )  # camL -> cam0
    cam_irR_rel_extrinsic_base = np.loadtxt(
        os.path.join(materials_root, "cam_irR_rel_extrinsic_base.txt")
    )  # camR -> cam0
    cam_irL_rel_extrinsic_hand = np.loadtxt(
        os.path.join(materials_root, "cam_irL_rel_extrinsic_hand.txt")
    )  # camL -> cam0
    cam_irR_rel_extrinsic_hand = np.loadtxt(
        os.path.join(materials_root, "cam_irR_rel_extrinsic_hand.txt")
    )  # camR -> cam0

    builder = scene.create_actor_builder()
    cam_mount = builder.build_kinematic(name="real_camera")
    if fixed_angle:
        # reproduce IJRR
        base_cam_rgb, base_cam_irl, base_cam_irr = create_realsense_d415(
            "real_camera_base", cam_mount, scene, cam_intrinsic_base, cam_ir_intrinsic_base
        )

    hand_cam_rgb, hand_cam_irl, hand_cam_irr = create_realsense_d415(
        "real_camera_hand", cam_mount, scene, cam_intrinsic_hand, cam_ir_intrinsic_hand
    )

    # Add lights
    if rand_env:
        ambient_light = np.random.rand(3)
        scene.set_ambient_light(ambient_light)
        scene.set_environment_map(get_random_env_file())

        # change light
        def lights_on():
            ambient_light = np.random.rand(3)
            scene.set_ambient_light(ambient_light)
            scene.set_environment_map(get_random_env_file())

            alight.set_color([0.0, 0.0, 0.0])

        def lights_off():
            ambient_light = np.random.rand(3) * 0.05
            scene.set_ambient_light(ambient_light)
            alight_color = np.random.rand(3) * np.array((60, 20, 20)) + np.array([30, 10, 10])
            scene.set_environment_map(get_random_env_file())

            alight.set_color(alight_color)

        def light_off_without_alight():
            alight.set_color([0.0, 0.0, 0.0])

    elif rand_lighting:
        ambient_light = np.random.rand(3)
        scene.set_ambient_light(ambient_light)
        height = np.random.rand() + 2
        light_1_color = np.random.rand(3) * 20 + 20
        light_2_color = np.random.rand(3) * 10 + 5
        light_3_color = np.random.rand(3) * 10 + 5
        light_4_color = np.random.rand(3) * 10 + 5
        light_5_color = np.random.rand(3) * 10 + 5
        plight1 = scene.add_point_light([-0.3, -0.3, height], light_1_color)
        plight2 = scene.add_point_light([2, -2, height], light_2_color)
        plight3 = scene.add_point_light([-2, 2, height], light_3_color)
        plight4 = scene.add_point_light([2, 2, height], light_4_color)
        plight5 = scene.add_point_light([-2, -2, height], light_5_color)

        # change light
        def lights_on():
            ambient_light = np.random.rand(3)
            scene.set_ambient_light(ambient_light)
            light_1_color = np.random.rand(3) * 20 + 20
            light_2_color = np.random.rand(3) * 10 + 5
            light_3_color = np.random.rand(3) * 10 + 5
            light_4_color = np.random.rand(3) * 10 + 5
            light_5_color = np.random.rand(3) * 10 + 5
            plight1.set_color(light_1_color)
            plight2.set_color(light_2_color)
            plight3.set_color(light_3_color)
            plight4.set_color(light_4_color)
            plight5.set_color(light_5_color)
            alight.set_color([0.0, 0.0, 0.0])

        def lights_off():
            ambient_light = np.random.rand(3) * 0.05
            scene.set_ambient_light(ambient_light)
            alight_color = np.random.rand(3) * np.array((60, 20, 20)) + np.array([30, 10, 10])
            light_1_color = np.random.rand(3) * 20 + 20
            light_2_color = np.random.rand(3) * 10 + 5
            light_3_color = np.random.rand(3) * 10 + 5
            light_4_color = np.random.rand(3) * 10 + 5
            light_5_color = np.random.rand(3) * 10 + 5
            plight1.set_color(light_1_color * 0.01)
            plight2.set_color(light_2_color * 0.01)
            plight3.set_color(light_3_color * 0.01)
            plight4.set_color(light_4_color * 0.01)
            plight5.set_color(light_5_color * 0.01)
            alight.set_color(alight_color)

        def light_off_without_alight():
            alight.set_color([0.0, 0.0, 0.0])

    else:
        scene.set_ambient_light([0.5, 0.5, 0.5])
        plight1 = scene.add_point_light([-0.3, -0.3, 2.5], [30, 30, 30])
        plight2 = scene.add_point_light([2, -2, 2.5], [10, 10, 10])
        plight3 = scene.add_point_light([-2, 2, 2.5], [10, 10, 10])
        plight4 = scene.add_point_light([2, 2, 2.5], [10, 10, 10])
        plight5 = scene.add_point_light([-2, -2, 2.5], [10, 10, 10])

        # change light
        def lights_on():
            scene.set_ambient_light([0.5, 0.5, 0.5])
            plight1.set_color([30, 30, 30])
            plight2.set_color([10, 10, 10])
            plight3.set_color([10, 10, 10])
            plight4.set_color([10, 10, 10])
            plight5.set_color([10, 10, 10])
            alight.set_color([0.0, 0.0, 0.0])

        def lights_off():
            p_scale = 4.0
            scene.set_ambient_light([0.03, 0.03, 0.03])
            plight1.set_color([0.3 * p_scale, 0.1 * p_scale, 0.1 * p_scale])
            plight2.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight3.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight4.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight5.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            alight.set_color([60.0, 20.0, 20.0])

        def light_off_without_alight():
            p_scale = 4.0
            scene.set_ambient_light([0.03, 0.03, 0.03])
            plight1.set_color([0.3 * p_scale, 0.1 * p_scale, 0.1 * p_scale])
            plight2.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight3.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight4.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            plight5.set_color([0.1 * p_scale, 0.03 * p_scale, 0.03 * p_scale])
            alight.set_color([0.0, 0.0, 0.0])

    mount_T = t3d.quaternions.quat2mat((-0.5, 0.5, 0.5, -0.5))
    fov = np.random.uniform(1.3, 2.0)
    tex = cv2.imread(os.path.join(materials_root, "d415-pattern-sq.png"))

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        return result

    tmp_idx = np.random.randint(1e8)
    if rand_pattern:
        angle = np.random.uniform(-90, 90)
        tex_tmp = rotate_image(tex, angle)
        cv2.imwrite(os.path.join(materials_root, f"d415-pattern-sq-tmp-{tmp_idx:08d}.png"), tex_tmp)

        alight = scene.add_active_light(
            pose=Pose([0.4, 0, 0.8]),
            # pose=Pose(cam_mount.get_pose().p, apos),
            color=[0, 0, 0],
            fov=fov,
            tex_path=os.path.join(materials_root, f"d415-pattern-sq-tmp-{tmp_idx:08d}.png"),
        )
    else:
        alight = scene.add_active_light(
            pose=Pose([0.4, 0, 0.8]),
            # pose=Pose(cam_mount.get_pose().p, apos),
            color=[0, 0, 0],
            fov=fov,
            tex_path=os.path.join(materials_root, "d415-pattern-sq.png"),
        )

    cam_extrinsic_list = np.load(os.path.join(materials_root, "cam_db_neoneo.npy"))
    if fixed_angle:
        assert num_views <= cam_extrinsic_list.shape[0]
    else:
        # Obtain random camera poses around a specific center location
        obj_center = np.array([0.425, 0, 0])
        alpha_range = [0, 2 * np.pi]
        theta_range = [0.01, np.pi * 3 / 8]
        radius_range = [0.8, 1.2]
        angle_list = [
            (alpha, theta, radius)
            for alpha in np.linspace(alpha_range[0], alpha_range[1], 50)
            for theta in np.linspace(theta_range[0], theta_range[1], 10)
            for radius in np.linspace(radius_range[0], radius_range[1], 10)
        ]
        angle_choices = random.choices(angle_list, k=num_views)

    # set scene layout
    if primitives:
        num_asset = random.randint(PRIMITIVE_MIN, PRIMITIVE_MAX)
        primitive_info = {}
        for i in range(num_asset):
            info = load_random_primitives(scene, renderer=renderer, idx=i)
            primitive_info.update(info)
    elif primitives_v2:
        num_asset = random.randint(PRIMITIVE_MIN, PRIMITIVE_MAX)
        primitive_info = {}
        for i in range(num_asset):
            info = load_random_primitives_v2(scene, renderer=renderer, idx=i)
            primitive_info.update(info)
    else:
        if not os.path.exists(os.path.join(SCENE_DIR, f"{scene_id}/input.json")):
            logger.warning(f"{SCENE_DIR}/{scene_id}/input.json not exists.")
            return
        world_js = json.load(open(os.path.join(SCENE_DIR, f"{scene_id}/input.json"), "r"))
        assets = world_js.keys()
        poses_world = [None for _ in range(NUM_OBJECTS)]
        extents = [None for _ in range(NUM_OBJECTS)]
        scales = [None for _ in range(NUM_OBJECTS)]
        obj_ids = []
        object_names = []

        for obj_name in assets:
            load_obj(
                scene,
                obj_name,
                renderer=renderer,
                pose=sapien.Pose.from_transformation_matrix(world_js[obj_name]),
                is_kinematic=True,
                material_name="kuafu_material_new2",
            )
            obj_id = OBJECT_NAMES.index(obj_name)
            obj_ids.append(obj_id)
            object_names.append(obj_name)
            poses_world[obj_id] = world_js[obj_name]
            extents[obj_id] = np.array(
                [
                    float(OBJECT_DB[obj_name]["x_dim"]),
                    float(OBJECT_DB[obj_name]["y_dim"]),
                    float(OBJECT_DB[obj_name]["z_dim"]),
                ],
                dtype=np.float32,
            )
            scales[obj_id] = np.ones(3)
        obj_info = {
            "poses_world": poses_world,
            "extents": extents,
            "scales": scales,
            "object_ids": obj_ids,
            "object_names": object_names,
        }

    for view_id in range(num_views):
        folder_path = os.path.join(target_root, f"{scene_id}-{view_id}")
        os.makedirs(folder_path, exist_ok=True)
        if fixed_angle:
            cam_mount.set_pose(cv2ex2pose(cam_extrinsic_list[view_id]))
            apos = cam_mount.get_pose().to_transformation_matrix()[:3, :3] @ mount_T
            apos = t3d.quaternions.mat2quat(apos)
            alight.set_pose(Pose(cam_mount.get_pose().p, apos))

        else:
            # Obtain random camera extrinsic
            alpha, theta, radius = angle_choices[view_id]
            cam_extrinsic = spherical_pose(center=obj_center, radius=radius, alpha=alpha, theta=theta)
            cam_mount.set_pose(sapien.Pose.from_transformation_matrix(cam_extrinsic))

            apos = cam_mount.get_pose().to_transformation_matrix()[:3, :3] @ mount_T
            apos = t3d.quaternions.mat2quat(apos)
            alight.set_pose(Pose(cam_mount.get_pose().p, apos))
        if view_id == 0 and fixed_angle:
            # reproduce IJRR
            cam_rgb = base_cam_rgb
            cam_irl = base_cam_irl
            cam_irr = base_cam_irr
        else:
            cam_rgb = hand_cam_rgb
            cam_irl = hand_cam_irl
            cam_irr = hand_cam_irr

        if not os.path.exists(os.path.join(folder_path, f"{spp:04d}_irR_kuafu_half_no_ir.png")):
            # begin rendering
            lights_on()
            scene.update_render()
            # Render mono-view RGB camera
            cam_rgb.take_picture()
            p = cam_rgb.get_color_rgba()
            plt.imsave(os.path.join(folder_path, f"{spp:04d}_rgb_kuafu.png"), p)

            # Render multi-view RGB camera
            cam_irl.take_picture()
            p = cam_irl.get_color_rgba()
            plt.imsave(os.path.join(folder_path, f"{spp:04d}_rgbL_kuafu.png"), p)

            cam_irr.take_picture()
            p = cam_irr.get_color_rgba()
            plt.imsave(os.path.join(folder_path, f"{spp:04d}_rgbR_kuafu.png"), p)
            plt.close("all")

            lights_off()
            scene.update_render()

            # Render multi-view IR camera
            cam_irl.take_picture()
            p_l = cam_irl.get_color_rgba()
            cam_irr.take_picture()
            p_r = cam_irr.get_color_rgba()
            p_l = (p_l[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_r = (p_r[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_l = cv2.cvtColor(p_l, cv2.COLOR_RGB2GRAY)
            p_r = cv2.cvtColor(p_r, cv2.COLOR_RGB2GRAY)
            p_l = cv2.GaussianBlur(p_l, (3, 3), 1)
            p_r = cv2.GaussianBlur(p_r, (3, 3), 1)
            irl = p_l[::2, ::2]
            irr = p_r[::2, ::2]
            cv2.imwrite(os.path.join(folder_path, f"{spp:04d}_irL_kuafu_half.png"), irl)
            cv2.imwrite(os.path.join(folder_path, f"{spp:04d}_irR_kuafu_half.png"), irr)

            light_off_without_alight()
            scene.update_render()

            # Render multi-view IR camera without pattern
            cam_irl.take_picture()
            p_l = cam_irl.get_color_rgba()
            cam_irr.take_picture()
            p_r = cam_irr.get_color_rgba()
            p_l = (p_l[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_r = (p_r[..., :3] * 255).clip(0, 255).astype(np.uint8)
            p_l = cv2.cvtColor(p_l, cv2.COLOR_RGB2GRAY)
            p_r = cv2.cvtColor(p_r, cv2.COLOR_RGB2GRAY)
            p_l = cv2.GaussianBlur(p_l, (3, 3), 1)
            p_r = cv2.GaussianBlur(p_r, (3, 3), 1)
            no_irl = p_l[::2, ::2]
            no_irr = p_r[::2, ::2]
            cv2.imwrite(os.path.join(folder_path, f"{spp:04d}_irL_kuafu_half_no_ir.png"), no_irl)
            cv2.imwrite(os.path.join(folder_path, f"{spp:04d}_irR_kuafu_half_no_ir.png"), no_irr)
        else:
            logger.info(f"skip {folder_path} rendering")

        if fixed_angle:
            cam_extrinsic = cam_extrinsic_list[view_id]
        else:
            cam_extrinsic = pose2cv2ex(cam_extrinsic)
        if view_id == 0 and fixed_angle:
            # reproduce IJRR
            cam_irL_extrinsic = np.linalg.inv(np.linalg.inv(cam_extrinsic) @ cam_irL_rel_extrinsic_base)
            cam_irR_extrinsic = np.linalg.inv(np.linalg.inv(cam_extrinsic) @ cam_irR_rel_extrinsic_base)
            cam_intrinsic = cam_intrinsic_base
            cam_ir_intrinsic = cam_ir_intrinsic_base
        else:
            cam_irL_extrinsic = np.linalg.inv(np.linalg.inv(cam_extrinsic) @ cam_irL_rel_extrinsic_hand)
            cam_irR_extrinsic = np.linalg.inv(np.linalg.inv(cam_extrinsic) @ cam_irR_rel_extrinsic_hand)
            cam_intrinsic = cam_intrinsic_hand
            cam_ir_intrinsic = cam_ir_intrinsic_hand

        # Save scene info
        scene_info = {
            "extrinsic": cam_extrinsic,
            "extrinsic_l": cam_irL_extrinsic,
            "extrinsic_r": cam_irR_extrinsic,
            "intrinsic": cam_intrinsic,
            "intrinsic_l": cam_ir_intrinsic,
            "intrinsic_r": cam_ir_intrinsic,
        }
        if primitives or primitives_v2:
            scene_info["primitives"] = primitive_info
        else:
            scene_info.update(obj_info)

        with open(os.path.join(folder_path, "meta.pkl"), "wb") as f:
            pickle.dump(scene_info, f)

        logger.info(f"finish {folder_path} rendering")
    if rand_pattern:
        os.remove(os.path.join(materials_root, f"d415-pattern-sq-tmp-{tmp_idx:08d}.png"))
    scene = None


def render_gt_depth_label(
    sim: sapien.Engine,
    renderer: sapien.VulkanRenderer,
    scene_id,
    repo_root,
    target_root,
    spp,
    num_views,
    rand_pattern,
    fixed_angle,
    primitives,
    primitives_v2,
):
    materials_root = os.path.join(repo_root, "data_rendering/materials")

    # build scene
    # sim = sapien.Engine()
    # sim.set_log_level("err")
    #
    # renderer = sapien.VulkanRenderer(offscreen_only=True)
    # renderer.set_log_level("err")
    # sim.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene_config.solver_iterations = 25
    scene_config.solver_velocity_iterations = 2
    scene_config.enable_pcm = False
    scene_config.default_restitution = 0
    scene_config.default_dynamic_friction = 0.5
    scene_config.default_static_friction = 0.5
    scene = sim.create_scene(scene_config)

    ground_material = renderer.create_material()
    ground_material.base_color = np.array([10, 10, 10, 256]) / 256
    ground_material.specular = 0.5
    scene.add_ground(-2.0, render_material=ground_material)
    scene.set_timestep(1 / 240)

    # Add camera
    cam_intrinsic_base = np.loadtxt(os.path.join(materials_root, "cam_intrinsic_base.txt"))
    cam_ir_intrinsic_base = np.loadtxt(os.path.join(materials_root, "cam_ir_intrinsic_base.txt"))
    cam_intrinsic_hand = np.loadtxt(os.path.join(materials_root, "cam_intrinsic_hand.txt"))
    cam_ir_intrinsic_hand = np.loadtxt(os.path.join(materials_root, "cam_ir_intrinsic_hand.txt"))
    cam_irL_rel_extrinsic_base = np.loadtxt(
        os.path.join(materials_root, "cam_irL_rel_extrinsic_base.txt")
    )  # camL -> cam0
    cam_irR_rel_extrinsic_base = np.loadtxt(
        os.path.join(materials_root, "cam_irR_rel_extrinsic_base.txt")
    )  # camR -> cam0
    cam_irL_rel_extrinsic_hand = np.loadtxt(
        os.path.join(materials_root, "cam_irL_rel_extrinsic_hand.txt")
    )  # camL -> cam0
    cam_irR_rel_extrinsic_hand = np.loadtxt(
        os.path.join(materials_root, "cam_irR_rel_extrinsic_hand.txt")
    )  # camR -> cam0

    builder = scene.create_actor_builder()
    cam_mount = builder.build_kinematic(name="real_camera")
    if fixed_angle:
        # reproduce IJRR
        base_cam_rgb, base_cam_irl, base_cam_irr = create_realsense_d415(
            "real_camera_base", cam_mount, scene, cam_intrinsic_base, cam_ir_intrinsic_base
        )

    hand_cam_rgb, hand_cam_irl, hand_cam_irr = create_realsense_d415(
        "real_camera_hand", cam_mount, scene, cam_intrinsic_hand, cam_ir_intrinsic_hand
    )

    table_pose_np = np.loadtxt(os.path.join(repo_root, "data_rendering/materials/optical_table/pose.txt"))
    table_pose = sapien.Pose(table_pose_np[:3], table_pose_np[3:])

    load_table_vk(scene, os.path.join(repo_root, "data_rendering/materials/optical_table"), table_pose)
    # Add lights
    scene.set_ambient_light([0.5, 0.5, 0.5])

    # set scene layout
    if primitives:
        # Load primitives from saved meta.pkl
        main_meta_path = os.path.join(target_root, f"{scene_id}-0", "meta.pkl")
        main_meta_info = load_pickle(main_meta_path)
        primitives_info = main_meta_info["primitives"]
        num_asset = len(primitives_info.keys())
        for i in range(num_asset):
            primitive = primitives_info[f"obj_{i}"]
            load_random_primitives_from_info(scene, renderer=renderer, idx=i, primitive_info=primitive)
    elif primitives_v2:
        # Load primitives from saved meta.pkl
        main_meta_path = os.path.join(target_root, f"{scene_id}-0", "meta.pkl")
        main_meta_info = load_pickle(main_meta_path)
        primitives_info = main_meta_info["primitives"]
        num_asset = len(primitives_info.keys())
        for i in range(num_asset):
            primitive = primitives_info[f"obj_{i}"]
            load_random_primitives_from_info(scene, renderer=renderer, idx=i, primitive_info=primitive)

    else:
        if not os.path.exists(os.path.join(SCENE_DIR, f"{scene_id}/input.json")):
            logger.warning(f"{SCENE_DIR}/{scene_id}/input.json not exists.")
            return
        world_js = json.load(open(os.path.join(SCENE_DIR, f"{scene_id}/input.json"), "r"))
        assets = world_js.keys()
        actors = []
        for obj_name in assets:
            ac = load_obj_vk(
                scene,
                obj_name,
                pose=sapien.Pose.from_transformation_matrix(world_js[obj_name]),
                is_kinematic=True,
            )
            actors.append(ac)

        # Mapping id to name
        seg_id_to_sapien_name = {actor.get_id(): actor.get_name() for actor in scene.get_all_actors()}
        seg_id_to_obj_name = {actor.get_id(): asset for actor, asset in zip(actors, assets)}

    for view_id in range(num_views):
        folder_path = os.path.join(target_root, f"{scene_id}-{view_id}")

        # Load meta info
        meta_info = load_pickle(os.path.join(folder_path, "meta.pkl"))

        cam_extrinsic = meta_info["extrinsic"]
        cam_mount.set_pose(cv2ex2pose(cam_extrinsic))

        # Obtain main-view RGB depth
        scene.update_render()
        if view_id == 0 and fixed_angle:
            # reproduce IJRR
            cam_rgb = base_cam_rgb
            cam_irl = base_cam_irl
            cam_irr = base_cam_irr
        else:
            cam_rgb = hand_cam_rgb
            cam_irl = hand_cam_irl
            cam_irr = hand_cam_irr
        cam_rgb.take_picture()
        p = cam_rgb.get_color_rgba()
        plt.imsave(os.path.join(folder_path, f"0000_rgb_vulkan.png"), p)
        pos = cam_rgb.get_float_texture("Position")
        depth = -pos[..., 2]
        depth = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(folder_path, f"depth.png"), depth)
        vis_depth = visualize_depth(depth)
        cv2.imwrite(os.path.join(folder_path, f"depth_colored.png"), vis_depth)

        # Obtain multi-vierw IR depth
        # scene.update_render()
        cam_irl.take_picture()
        pos = cam_irl.get_float_texture("Position")
        depth = -pos[..., 2]
        depth = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(folder_path, f"depthL.png"), depth)
        vis_depth = visualize_depth(depth)
        cv2.imwrite(os.path.join(folder_path, f"depthL_colored.png"), vis_depth)

        # scene.update_render()
        cam_irr.take_picture()
        pos = cam_irr.get_float_texture("Position")
        depth = -pos[..., 2]
        depth = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(folder_path, f"depthR.png"), depth)
        vis_depth = visualize_depth(depth)
        cv2.imwrite(os.path.join(folder_path, f"depthR_colored.png"), vis_depth)

        if not (primitives or primitives_v2):
            # render label image
            obj_segmentation = cam_rgb.get_uint32_texture("Segmentation")[..., 1]

            assert not np.any(obj_segmentation > 255)

            # Mapping seg_id to obj_id, as a semantic label image.
            seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
            seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
            obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

            # Semantic labels
            sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
            sem_image = Image.fromarray(sem_labels)
            sem_image.save(os.path.join(folder_path, "label.png"))

            sem_labels_with_color = COLOR_PALETTE[sem_labels]
            sem_image_with_color = Image.fromarray(sem_labels_with_color.astype("uint8"))
            sem_image_with_color.save(os.path.join(folder_path, "label2.png"))

            obj_segmentation = cam_irl.get_uint32_texture("Segmentation")[..., 1]

            assert not np.any(obj_segmentation > 255)

            # Mapping seg_id to obj_id, as a semantic label image.
            seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
            seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
            obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

            # Semantic labels
            sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
            sem_image = Image.fromarray(sem_labels)
            sem_image.save(os.path.join(folder_path, "labelL.png"))

            sem_labels_with_color = COLOR_PALETTE[sem_labels]
            sem_image_with_color = Image.fromarray(sem_labels_with_color.astype("uint8"))
            sem_image_with_color.save(os.path.join(folder_path, "labelL2.png"))

            obj_segmentation = cam_irr.get_uint32_texture("Segmentation")[..., 1]

            assert not np.any(obj_segmentation > 255)

            # Mapping seg_id to obj_id, as a semantic label image.
            seg_id_to_fused_name = {k: seg_id_to_obj_name.get(k, v) for k, v in seg_id_to_sapien_name.items()}
            seg_id_to_obj_id = get_seg_id_to_obj_id(seg_id_to_fused_name)
            obj_ids = np.unique(seg_id_to_obj_id[seg_id_to_obj_id < NUM_OBJECTS])

            # Semantic labels
            sem_labels = (seg_id_to_obj_id[obj_segmentation]).astype("uint8")
            sem_image = Image.fromarray(sem_labels)
            sem_image.save(os.path.join(folder_path, "labelR.png"))

            sem_labels_with_color = COLOR_PALETTE[sem_labels]
            sem_image_with_color = Image.fromarray(sem_labels_with_color.astype("uint8"))
            sem_image_with_color.save(os.path.join(folder_path, "labelR2.png"))

        logger.info(f"finish {folder_path} gt depth and seg")
    scene = None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--target-root", type=str, required=True)
    parser.add_argument("--scene", "-s", type=str, required=True)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--nv", type=int, default=21)
    parser.add_argument("--rand-pattern", action="store_true")
    parser.add_argument("--fixed-angle", action="store_true")
    parser.add_argument("--primitives", action="store_true", help="use primitives")
    parser.add_argument("--primitives-v2", action="store_true", help="use primitives v2")
    args = parser.parse_args()

    assert not (args.primitives and args.primitives_v2), "primitives and v2 cannot be True in one run"
    render_scene(
        args.scene,
        repo_root=args.repo_root,
        target_root=args.target_root,
        spp=args.spp,
        num_views=args.nv,
        rand_pattern=args.rand_pattern,
        fixed_angle=args.fixed_angle,
        primitives=args.primitives,
        primitives_v2=args.primitives_v2,
    )
    render_gt_depth_label(
        args.scene,
        repo_root=args.repo_root,
        target_root=args.target_root,
        spp=args.spp,
        num_views=args.nv,
        rand_pattern=args.rand_pattern,
        fixed_angle=args.fixed_angle,
        primitives=args.primitives,
        primitives_v2=args.primitives_v2,
    )
