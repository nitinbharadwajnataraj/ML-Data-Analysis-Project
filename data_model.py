# Preloaded dictionaries containing target, min, max values, and factors for each process
data_model = {
    "Cube": {
        "Cube_Diameter": {"target_value": 30, "min_value": 28, "max_value": 32, "min_eq_target": False,
                          "factor": 1.0},
        "Hole_Depth": {"target_value": 40, "min_value": 37, "max_value": 43, "min_eq_target": False,
                       "factor": 1.0}
    },
    "Cylinder": {
        "Cylinder_Diameter": {"target_value": 40, "min_value": 38.5, "max_value": 41.5, "min_eq_target": False,
                              "factor": 1.0},
        "Cylinder_Height": {"target_value": 30, "min_value": 27.5, "max_value": 32.5, "min_eq_target": False,
                            "factor": 1.0}
    },
    "Alignment": {
        "Joining_Force": {"target_value": 10, "min_value": 9, "max_value": 11, "min_eq_target": False,
                          "factor": 1.0}
    }
}
