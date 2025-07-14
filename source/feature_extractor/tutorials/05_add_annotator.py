import os
import asyncio
import omni.replicator.core as rep
import open3d as o3d
import numpy as np

async def test_pointcloud():
    # Pointcloud only capture prims with valid semantics
    cube = rep.create.cube(semantics=[("class", "cube")])
    camera = rep.create.camera()
    render_product = rep.create.render_product(camera, (1024, 512))

    pointcloud_anno = rep.annotators.get("pointcloud")
    pointcloud_anno.attach(render_product)

    # Camera positions to capture the cube
    camera_positions = [(500, 500, 0), (-500, -500, 0), (500, 0, 500), (-500, 0, -500)]

    with rep.trigger.on_frame(num_frames=len(camera_positions)):
        with camera:
            rep.modify.pose(position=rep.distribution.sequence(camera_positions), look_at=cube)  # make the camera look at the cube

    # Accumulate points
    points = []
    points_rgb = []
    points_seg = []
    for _ in range(len(camera_positions)):
        await rep.orchestrator.step_async()

        pc_data = pointcloud_anno.get_data()
        # {
        #     'data': array([...], shape=(<num_points>, 3), dtype=float32),
        #     'info': {
        #         'pointNormals': [ 0.000e+00 1.00e+00 -1.5259022e-05 ... 0.00e+00 -1.5259022e-05 1.00e+00], shape=(<num_points> * 4), dtype=float32),
        #         'pointRgb': [241 240 241 ... 11  12 255], shape=(<num_points> * 4), dtype=uint8),
        #         'pointSemantic': [2 2 2 ... 2 2 2], shape=(<num_points>), dtype=uint8),
        #
        #     }
        # }
        points.append(pc_data["data"])
        points_rgb.append(pc_data["info"]["pointRgb"].reshape(-1, 4)[:, :3])
        points_seg.append(pc_data["info"]["pointSemantic"])

    # Output pointcloud as .ply file
    ply_out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "out")
    os.makedirs(ply_out_dir, exist_ok=True)
    print(f"make directory : {ply_out_dir}")

    pc_data = np.concatenate(points)
    pc_rgb = np.concatenate(points_rgb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_data)
    pcd.colors = o3d.utility.Vector3dVector(pc_rgb)
    o3d.io.write_point_cloud(os.path.join(ply_out_dir, "pointcloud.ply"), pcd)

asyncio.ensure_future(test_pointcloud())