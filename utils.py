import yaml

def getConfig(configFilePath="config.yaml"):
    with open(configFilePath, "r") as cfgFile:
        try:
            config = yaml.safe_load(cfgFile)
        except yaml.YAMLError as exc:
            print(exc)
    return config
