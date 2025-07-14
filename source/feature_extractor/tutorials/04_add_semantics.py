import omni.replicator.core as rep

with rep.new_layer():

    # Add Default Light
    distance_light = rep.create.light(rotation=(315,0,0), intensity=3000, light_type="distant")

    # Defining a plane to place the avocado
    plane = rep.create.plane(scale=100, visible=True)

    # Defining the avocado starting from the NVIDIA residential provided assets. Position and semantics of this asset are modified.
    AVOCADO = 'omniverse://localhost/NVIDIA/Assets/ArchVis/Residential/Food/Fruit/Avocado01.usd'
    avocado = rep.create.from_usd(AVOCADO)
    with avocado:
        rep.modify.semantics([('class', 'avocado')])
        rep.modify.pose(
                position=(0, 0, 0),
                rotation=(-90,-45, 0)
                )

    # Setup camera and attach it to render product
    camera = rep.create.camera(focus_distance=30)
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))

    writer = rep.WriterRegistry.get("BasicWriter")
    
    # 저장할 경로와 데이터 종류를 지정합니다.
    # 여기서는 RGB 이미지와 Semantic Segmentation(의미 정보) 데이터를 저장합니다.
    writer.initialize(
        output_dir="_output", # "_output" 이라는 폴더에 저장
        rgb=True,
        semantic_segmentation=True
    )

    # render_product에 writer를 연결(attach)합니다.
    # 이 부분이 렌더링 결과와 파일 저장을 이어주는 핵심입니다.
    writer.attach([render_product])

    # 카메라가 무조건 아보카도의 중심점을 바라보도록  설정할 수 있음 (look_at 매개변수)
    with rep.trigger.on_frame(num_frames=30):
        with camera:
            rep.modify.pose(position=rep.distribution.uniform((-10, 10, 20), (10, 20, 30)), look_at=avocado)