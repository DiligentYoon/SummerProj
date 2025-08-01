import omni.replicator.core as rep

# 위치를 Randomization 할 카메라 2대 생성 
camera1_pos = [(0, 100, 500),(500, 100, 500),(-500, 100, 500),]
camera2_pos = [(0, 500, 0),(500, 500, 0),(-500, 500, 0),]

with rep.new_layer():

    # Add Default Light
    distance_light = rep.create.light(rotation=(315,0,0), intensity=3000, light_type="distant")

    # Object 생성
    cube = rep.create.cube(position=(200, 0, 0), semantics=[('class', 'cube')])
    sphere = rep.create.sphere(position=(0, 0, 0), semantics=[('class', 'sphere')])
    cone = rep.create.cone(position=(-200, 0, 0), semantics=[('class', 'cone')])

    # 카메라 2대 생성
    camera = rep.create.camera(position=camera1_pos[0], look_at=(0,0,0))
    camera2 = rep.create.camera(position=camera2_pos[0], look_at=(0,0,0))

    # 카메라의 3프레임 마다 두 카메라의 위치 랜덤 변경 in list 후보군
    with rep.trigger.on_frame(num_frames=3):
        with camera:
            rep.modify.pose(look_at=(0,0,100 ), position=rep.distribution.sequence(camera1_pos))
        with camera2:
            rep.modify.pose(look_at=(0,0,0), position=rep.distribution.sequence(camera2_pos))

# Will render 512x512 images and 320x240 images
render_product = rep.create.render_product(camera, (512, 512))
render_product2 = rep.create.render_product(camera2, (320, 240))

# writer 생성 : 모든 데이터 형태 저장
basic_writer = rep.WriterRegistry.get("BasicWriter")
basic_writer.initialize(
    output_dir=f"~/replicator_examples/multi_render_product/basic",
    rgb=True,
    bounding_box_2d_loose=True,
    bounding_box_2d_tight=True,
    bounding_box_3d=True,
    distance_to_camera=True,
    distance_to_image_plane=True,
    instance_segmentation=True,
    normals=True,
    semantic_segmentation=True,
)
# Attach render_product to the writer
basic_writer.attach([render_product, render_product2])
# Run the simulation graph
rep.orchestrator.run()