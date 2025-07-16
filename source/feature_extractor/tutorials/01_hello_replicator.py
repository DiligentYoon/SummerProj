import omni.replicator.core as rep

with rep.new_layer():
    # 카메라 스폰
    camera = rep.create.camera(position=(0, 0, 1000))
    # 카메라와 replicator 간의 상호작용 브릿지인 render 생성
    render_product = rep.create.render_product(camera, (1024, 1024))

    # object 스폰
    torus = rep.create.torus(semantics=[('class', 'torus')] , position=(0, 1 , 1))

    sphere = rep.create.sphere(semantics=[('class', 'sphere')], position=(0, 2, 1))

    cube = rep.create.cube(semantics=[('class', 'cube')],  position=(0, 3 , 1) )

    # 입력받은 카메라 frame마다 domain randomization 수행
    with rep.trigger.on_frame(num_frames=10):
        with rep.create.group([torus, sphere, cube]):
            rep.modify.pose(
                position=rep.distribution.uniform((-100, -100, -100), (200, 200, 200)),
                scale=rep.distribution.uniform(1, 2))

    # 특정 데이터 포맷으로 데이터셋을 저장할 writer 스폰
    writer = rep.WriterRegistry.get("BasicWriter")

    # output 폴더 및 저장하고자 하는 데이터 포맷 생성
    writer.initialize( output_dir="_output", rgb=True,   bounding_box_2d_tight=True)

    # renderer에 부착
    writer.attach([render_product])

    # 프리뷰 실행
    rep.orchestrator.run()