"""Prepare an opt-in native hidden-contact profile on the configured output disk."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    config=json.loads(a.environment_config.read_text(encoding='utf-8-sig'))
    profile=json.loads(Path(config['profile']).read_text(encoding='utf-8-sig'))
    if profile.get('hooks',{}).get('native_enemy_forecast')!='th14-native-enemy-v1':
        raise ValueError('Native enemy forecasts are required')
    a.output.mkdir(parents=True,exist_ok=False)
    profile['hooks']['native_hidden_enemy_contacts']='th14-hidden-enemy-contact-v1'
    profile_path=a.output/'th14-profile-hidden-contact-v1.json'
    profile_path.write_text(json.dumps(profile,ensure_ascii=False,indent=2),encoding='utf-8')
    config['profile']=str(profile_path)
    config_path=a.output/'environment-config-hidden-contact-v1.json'
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(config_path)


if __name__=='__main__':main()
