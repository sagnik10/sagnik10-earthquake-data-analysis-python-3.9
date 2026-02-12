import xsection as xs
from xsection import CompositeSection
from xsection.annulus import Circle
import veux

# test_stress = lambda s: max(s) < 0.9*50*ksi
# test_stress = lambda s: min(s) > -0.9*50*ksi
# test_strain = lambda s: max(s) < 0.9*0.003
# test_strain = lambda s: min(s) > -0.9*0.003


if __name__ == "__main__":

    a = CompositeSection(patches=[
            Circle(30.0, z=1),
            Circle(20.0, z=1).translate([0, 10]),
            *Circle(1.0, z=2).replicate([-3, 3], [3, 0], 10, center=[0,0]),
    ])

    veux.render(a.shape)
    veux.render(a.model, field=a.torsion_warping())
    veux.render(a.model, field=a.flexure_warping())

    m = {
        1: {"Fy": 50.0, "E": 29000.0, "type": "J2"},
        2: {"Fc":  4.0, "E": 29000.0, "type": "DruckerPrager"},
    }

    xs.ResultantInteraction(a, m, ["M", "N"],
                            shear=True,
                            min_strain=0,
                            max_strain=0.003,
                            min_stress=-50,
                            max_stress=50
    )

    for fiber in a.create_fibers(material=1):
        pass



    a.elastic.A
    a.elastic.center
    a.plastic.center
    a.torsion.center
    a.flexure.center
    a.trefftz.center
    a.warping.

    a.elastic.A
    a.elastic.Iz
    a.elastic.Iyz
    a.elastic.summary()
