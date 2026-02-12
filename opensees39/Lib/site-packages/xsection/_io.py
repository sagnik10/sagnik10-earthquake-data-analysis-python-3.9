
import textwrap

_SCHEMA = {
    "type": {"type": "string", "enum": ["Polygon", "Standard", "Composite"]},
    "shape": {
        "type": "string",
        "enum": [
            "Rectangle",
            "Circle",
            "Rebar"
        ],
    },
    "shapes": {
        "type": "array",
        "items": {"$ref": "#/definitions/Shape"},
        "minItems": 1,
    },
    "parameters": {
        "type": "object",
        "additionalProperties": True,
    }
}

def _from_dict(data):

    stype = data["type"]
    if stype == "Polygon":
        from xsection.polygon import PolygonSection

        return PolygonSection.from_dict(data)

    elif stype == "Standard":
        from xsection import library 
        cls = getattr(library, data["shape"])
        return cls(**data["parameters"])

    elif stype == "Composite":
        shapes = [
            _from_dict(item) for item in data["shapes"]
        ]

def write(obj, file):
    """Write a section object to a file."""
    import json
    from xsection import CompositeSection

    if isinstance(obj, CompositeSection):
        data = obj.to_dict()
    else:
        data = obj.to_dict()

    file.write(json.dumps(data, indent=4))



def export_fedeas(shape, type, name=None, material=None):
    if name is None:
        name = "ImportSection"

    if type == "Elastic":
        # Export elastic properties
        pass

    elif type == "General":
        s = f"""
        function SecData = {name}(MatData, Units)
        SecData = struct();
        """
        for i,fiber in enumerate(shape.create_fibers()):
            s += f"""
            SecData.Fibers{{{i+1}}}.r    = [0, {fiber['y']}*Units.inch, {fiber['z']}*Units.inch];
            SecData.Fibers{{{i+1}}}.Area = {fiber['area']}*Units.inch^2;
            SecData.Fibers{{{i+1}}}.Warp = [
                {float(fiber['warp'][0][0])}, {float(fiber['warp'][0][1])}, {float(fiber['warp'][0][2])};
                {float(fiber['warp'][1][0])}, {float(fiber['warp'][1][1])}, {float(fiber['warp'][1][2])};
                {float(fiber['warp'][2][0])}, {float(fiber['warp'][2][1])}, {float(fiber['warp'][2][2])}
            ];
            SecData.Fibers{{{i+1}}}.MatData = MatData;
            """
        s += "end % function\n"

        return textwrap.dedent(s)
    else:
        raise ValueError(f"Unknown export type: {type}")
