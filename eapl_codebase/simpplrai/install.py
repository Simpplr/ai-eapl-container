import pip
import argparse

_all_ = [
    "django-basicauth==0.5.2",
    "django-rq==2.3.2",
    "pymongo==3.12.1",
    "django-environ",
    "freezegun"
]

_svr2brnch_map = {
    'emplay-dev': 'development',
    'emplay-qa': 'qa',
    'emplay-stg': 'stage',
    'emplay-prod': 'prod'
}

_deps_root = [
    "git+ssh://git@bitbucket.org/simpplr/eapl_nlp.git@",
    "git+ssh://git@bitbucket.org/simpplr/eapl.git@"
]


def install(packages):
    for package in packages:
        pip.main(['install', package])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server", required=True, help="Server for deployment")
    ap.add_argument("-u", "--username", required=False, help="Username for git clone of branch")
    ap.add_argument("-p", "--password", required=False, help="Password for git clone of branch")

    args = vars(ap.parse_args())
    server = args['server']
    username = args['username']
    password = args['password']

    deps = _deps_root.copy()
    if username and password:
        deps = [f"git+https://{username}:{password}@{d.split('git+ssh://git@')[1]}{_svr2brnch_map[server]}" for d in deps]
    else:
        deps = [f"{d}{_svr2brnch_map[server]}" for d in deps]
    install(_all_)
    install(deps)
