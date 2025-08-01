import asyncio
import omni.replicator.core as rep

async def test_pointcloud():
    with rep.new_layer():
        # Add Default Light
        distance_light = rep.create.light(rotation=(315,0,0), intensity=3000, light_type="distant")

        # Pointcloud only capture prims with valid semantics
        W, H = (960, 960)
        cube = rep.create.cube(position=(0, 0, 0), semantics=[("class", "cube")])
        camera = rep.create.camera(position=(200., 200., 200.), look_at=cube)
        render_product = rep.create.render_product(camera, (W, H))

        pointcloud_anno = rep.annotators.get("pointcloud")
        pointcloud_anno.attach(render_product)

        await rep.orchestrator.step_async()

        pc_data = pointcloud_anno.get_data()
        print(pc_data)
        # {
        #     'data': array([...], shape=(<num_points>, 3), dtype=float32),
        #     'info': {
        #         'pointNormals': [ 0.000e+00 1.00e+00 -1.5259022e-05 ... 0.00e+00 -1.5259022e-05 1.00e+00], shape=(<num_points> * 4), dtype=float32),
        #         'pointRgb': [241 240 241 ... 11  12 255], shape=(<num_points> * 4), dtype=uint8),
        #         'pointSemantic': [2 2 2 ... 2 2 2], shape=(<num_points>), dtype=uint8),
        #
        #     }
        # }

asyncio.ensure_future(test_pointcloud())