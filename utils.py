import yaml

def getConfig(configFilePath="config.yaml"):
    config = {}
    with open(configFilePath, "r") as cfgFile:
        try:
            config = yaml.safe_load(cfgFile)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def getTestConfig():
    return getConfig("test_config.yaml")