def get_modules_requirements():
    with open("requirements-modules.txt") as f:
        requirements_modules = f.read().splitlines()

    modules = {}

    for req in requirements_modules:
        if req.startswith("#"):
            tag = req.split(" ")[1]
            modules[tag] = []
        else:
            if len(req.strip()) > 0:
                modules[tag].append(req)

    return modules
