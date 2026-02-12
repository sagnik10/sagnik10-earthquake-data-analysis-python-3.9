OPTIONS = {
    "storage": ["Full", "Band", "Sparse", "Profile"],
}

SOLVERS = {
    "ProfileSPD": {"symmetric": [True],  "storage": "Profile", "driver": "ops_profile"},
    "Umfpack":    {"symmetric": [False], "storage": "Sparse",  "driver": "umfpack"},
    "Mumps":      {"symmetric": [False], "storage": "Sparse",  "driver": "mumps"},
    "SparseSPD":  {"symmetric": [True],  "storage": "Sparse",  "driver": "ops_sparse"},
    "BandGen":    {"symmetric": [False], "storage": "Band",    "driver": "DGBSV"},
    "BandSPD":    {"symmetric": [True],  "storage": "Band",    "driver": "DPBSV"},
    "FullGen":    {"symmetric": [False], "storage": "Full",    "driver": "DGESV"},
    "FullSPD":    {"symmetric": [True],  "storage": "Full",    "driver": "DPOSV"},
}


class LinearSystem:
    """
    LinearSystem("Mumps", symmetric=False)
    LinearSystem(detect=model)
    """

    def __init__(self,
                 first:     str = None,
                 solver:    str = None,
                 storage:   str = None,
                 symmetric: bool = None,
                 definite:  bool = None,
                 numberer:  str = None):
        
        if isinstance(first, str):
            if first.lower() in ["mumps", "umfpack"]:
                storage = "Sparse"
            elif "band" in first.lower():
                storage = "Band"
            elif "full" in first.lower():
                storage = "Full"
            elif "profile" in first.lower():
                storage = "Profile"
            elif "sparse" in first.lower():
                storage = "Sparse"
            else:
                raise ValueError(f"Unknown solver '{first}'")
            name = first
        
        elif first is None:
            if symmetric is True and storage in ["Full", "Band"]:
                name = "FullSPD" if storage == "Full" else "BandSPD"
            elif symmetric is True and storage == "Sparse":
                name = "SparseSPD"
            elif symmetric is False and storage in ["Full", "Band"]:
                name = "FullGen" if storage == "Full" else "BandGen"
            elif symmetric is False and storage == "Sparse":
                name = "Mumps"
            else:
                name = "Mumps"

        if symmetric is None:
            if name.lower() in ["mumps", "umfpack"]:
                symmetric = False
            else:
                symmetric = True


        if numberer is None:
            if storage.lower() in ["band", "profile"]:
                numberer = "RCM"
            elif storage.lower() in ["sparse"]:
                numberer = "AMD"
            else:
                numberer = "Plain"

        self.name = name
        self.solver = solver
        self.storage = storage
        self.numberer = numberer
        self.symmetric = symmetric
        self.definite = definite

    def __repr__(self):
        return f"LinearSystem('{self.name}', symmetric={self.symmetric}, storage='{self.storage}', numberer='{self.numberer}', definite={self.definite})"