import omni.replicator.core as rep
import os, numpy as np, open3d as o3d

# ------------------------------------------------------------------ #
# 0. 기본 세팅
# ------------------------------------------------------------------ #
BASE_PATH = os.path.join(os.getcwd(), "Dataset/google_objects_usd")
GROUND    = "omniverse://localhost/NVIDIA/Assets/Scenes/Templates/Basic/display_riser.usd"
RESOLUTION = (960, 480)
SAMPLES_PER_OBJECT = 2
VIEWS_PER_SAMPLE   = 4
TOTAL_FRAMES       = SAMPLES_PER_OBJECT * VIEWS_PER_SAMPLE   # 트리거용
ASSET_LIST = {
    "Mug": [
        "025_mug/025_mug.usd",
        "Room_Essentials_Mug_White_Yellow/Room_Essentials_Mug_White_Yellow.usd",
    ],
    "Cube": [
        "010_potted_meat_can/010_potted_meat_can.usd",
        "003_cracker_box/003_cracker_box.usd",
    ],
    "Cylinder": [
        "006_mustard_bottle/006_mustard_bottle.usd",
        "021_bleach_cleanser/021_bleach_cleanser.usd",
    ],
}

# 객체(USD) 목록만 평탄화
OBJECT_PATHS = []
for rels in ASSET_LIST.values():
    OBJECT_PATHS.extend([os.path.join(BASE_PATH, r).replace("\\", "/") for r in rels])

# ------------------------------------------------------------------ #
# 1. 씬 기본 레이어
# ------------------------------------------------------------------ #
with rep.new_layer():

    # (a) 지면
    ground = rep.create.from_usd(GROUND, semantics=[("class", "ground")])
    with ground:
        rep.modify.pose(rotation=(90, 0, 0), position=(0, 0, 0))
        rep.physics.collider()

    # (b) 카메라 + 렌더 & 어노테이터
    cam = rep.create.camera()
    rp  = rep.create.render_product(cam, RESOLUTION)
    pc_anno = rep.annotators.get("pointcloud")
    pc_anno.attach(rp)

    # ------------------------------------------------------------------ #
    # 2. 랜덤라이저 함수 정의
    # ------------------------------------------------------------------ #
    def randomize_scene():
        # ① 대상 물체 1개 인스턴스화 & 지면 위 배치
        obj = rep.randomizer.instantiate(OBJECT_PATHS, size=1)
        with obj:
            rep.physics.collider(approximation_shape="convexDecomposition")
            rep.modify.semantics([("class", "object")])
            # rep.randomizer.scatter_2d(surface_prims=[ground], check_for_collisions=True)
            rep.modify.pose(
                position = rep.distribution.uniform((1, 1, 1), (5, 5, 3)),
                rotation = rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                scale    = rep.distribution.uniform(5.0, 7.0)
            )

        return obj.node          # 반환값은 선택 사항이지만 추적용

    def randomize_light():
        lights = rep.create.light(
                light_type="Sphere",
                temperature=rep.distribution.normal(6500, 500),
                intensity=rep.distribution.normal(35000, 5000),
                position=rep.distribution.uniform((-300, -300, -300), (300, 300, 300)),
                scale=rep.distribution.uniform(50, 100),
                count=1
            )
        return lights.node

    def randomize_camera():
        obj = rep.get.prims(semantics=["class", "object"])

        with cam:
            rep.modify.pose(
                position = rep.distribution.uniform((-5, -5, 1), (5, 5, 5)),
                look_at  = obj      # 아래 트리거에서 obj를 넘겨 받아 갱신해도 OK
            )
        return cam.node
    
    # Register
    rep.randomizer.register(randomize_scene)
    rep.randomizer.register(randomize_light)
    rep.randomizer.register(randomize_camera)

    # ------------------------------------------------------------------ #
    # 3. 트리거: ‘언제’ 랜덤화할지 지정
    # ------------------------------------------------------------------ #
    # ┌──────────── interval=VIEWS_PER_SAMPLE (=4) ────────────┐
    # │ frame 0 1 2 3 | 4 5 6 7 | …                           │
    # │        ↑scene↑          ↑scene↑ …  ← obj & light 갱신 │
    # └────────────────────────────────────────────────────────┘

    with rep.trigger.on_frame(num_frames=TOTAL_FRAMES, interval=VIEWS_PER_SAMPLE):
        rep.randomizer.randomize_scene()
        rep.randomizer.randomize_light()

    # 카메라는 매 프레임마다 갱신

    with rep.trigger.on_frame(num_frames=TOTAL_FRAMES):
        rep.randomizer.randomize_camera()

    # ------------------------------------------------------------------ #
    # 4. 실행 (Replicator Writer가 자동 저장 → 후처리는 기존 방식 재사용)
    # ------------------------------------------------------------------ #
    rep.orchestrator.run()