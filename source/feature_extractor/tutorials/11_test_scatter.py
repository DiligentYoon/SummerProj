import omni.graph.core as og
import omni.replicator.core as rep

# create a plane to sample on
plane_samp = rep.create.plane(scale=4, rotation=(20, 0, 0))
with plane_samp:
    rep.physics.collider()


def randomize_spheres():
    # create small spheres to sample inside the plane
    spheres = rep.create.sphere(scale=0.4, count=30)

    # randomize
    with spheres:
        rep.randomizer.scatter_2d(plane_samp, check_for_collisions=True)
        # Add color to small spheres
        rep.randomizer.color(colors=rep.distribution.uniform((0.2, 0.2, 0.2), (1, 1, 1)))
    return spheres.node


rep.randomizer.register(randomize_spheres)

with rep.trigger.on_frame(num_frames=10):
    rep.randomizer.randomize_spheres()