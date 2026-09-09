"""Keep notebook rendering portable."""

from mkdocs.plugins import event_priority


@event_priority(-100)
def on_config(config):
    # mknotebooks uses os.path.join for CSS URLs, which produces backslashes
    # on Windows. MkDocs expects URL separators on every platform.
    config["extra_css"] = [path.replace("\\", "/") for path in config["extra_css"]]
    return config
