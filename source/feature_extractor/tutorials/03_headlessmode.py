import omni.replicator.core as rep

with rep.new_layer():

    camera = rep.create.camera(position=(0, 0, 1000))

    sphere_light = rep.create.light(
        light_type="Sphere",
        temperature=rep.distribution.normal(6500, 500),
        intensity=rep.distribution.normal(35000, 5000),
        position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
        scale=rep.distribution.uniform(50, 100),
        count=2
    )

    render_product = rep.create.render_product(camera, (1024, 1024))

    torus = rep.create.torus(semantics=[('class', 'torus')] , position=(0, -200 , 100))

    sphere = rep.create.sphere(semantics=[('class', 'sphere')], position=(0, 100, 100))

    cube = rep.create.cube(semantics=[('class', 'cube')],  position=(100, -200 , 100) )

    with rep.trigger.on_frame(num_frames=10):
        with rep.create.group([torus, sphere, cube]):
            rep.modify.pose(
                position=rep.distribution.uniform((-100, -100, -100), (200, 200, 200)),
                scale=rep.distribution.uniform(0.1, 2))

    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")

    writer.initialize( output_dir="_output", rgb=True,   bounding_box_2d_tight=True)

    writer.attach([render_product])

    rep.orchestrator.run()