import yaml

def getConfig(config_path="./configs/config.yaml"):
    config = {}
    with open(config_path, "r") as cfgFile:
        try:
            config = yaml.safe_load(cfgFile)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def getTestConfig():
    return getConfig("./configs/test_config.yaml")
