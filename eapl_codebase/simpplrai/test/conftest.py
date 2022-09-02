from dataclasses import dataclass


@dataclass
class Secret:
    url: str = None
    username: str = None
    password: str = None
    version: str = None


SECRET = Secret()


def pytest_addoption(parser):
    parser.addoption("--endpoint", action="store")
    parser.addoption("--username", action="store")
    parser.addoption("--password", action="store")
    parser.addoption("--versionstr", action="store", default="1.0")


def pytest_configure(config):
    SECRET.url = config.option.endpoint
    SECRET.username = config.option.username
    SECRET.password = config.option.password
    SECRET.version = config.option.versionstr
