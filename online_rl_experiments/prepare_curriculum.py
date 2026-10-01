"""Build the LF curriculum specification from the completed calibration."""
import json
from pathlib import Path
from prepare_data import sha256


def main():
    directory = Path('results/bigmath-calibration-20260927')
    calibration = json.loads((directory/'config.json').read_text())
    summary = json.loads((directory/'three_pass/summary.json').read_text())
    bands = ['0_to_0.125','0.125_to_0.25','0.25_to_0.5','0.5_to_0.75','0.75_to_0.9','0.9_to_1']
    def keys(source, low, high):
        return [source+'/'+band for band in bands[low:high]]
    pools = [dict(name='big_math',questions=154,strata=keys('big_math',2,5)),
             dict(name='math',questions=102,strata=keys('math',1,5)),
             dict(name='cn_k12',questions=102,strata=keys('cn_k12',2,5)),
             dict(name='orca_math',questions=77,strata=keys('orca_math',3,5)),
             dict(name='competition',questions=51,strata=keys('olympiads',2,5)+keys('aops_forum',3,5)+keys('omnimath',1,5)),
             dict(name='exploration',questions=26,strata=list(sorted(calibration['populations'])))]
    strata = {k:dict(population=n,
                    mixed_fraction=summary['by_stratum'][k]['mixed_groups']/summary['by_stratum'][k]['prompts'],
                    truncation_rate=summary['by_stratum'][k]['truncation_rate'])
              for k,n in calibration['populations'].items()}
    spec = dict(version=1,data_sha256=calibration['dataset']['prepared_sha256'],
        calibration_sha256=sha256(directory/'three_pass/summary.json'),
        calibration_checkpoint=calibration['checkpoints']['three_pass'],
        refresh_after_updates=25,prior_groups=20,pools=pools,strata=strata,
        sampling='Without replacement within each update; shuffled stratum decks persist across updates.',
        phase_two='Keep pool quotas; weight stratum populations by smoothed mixed-group fraction times (1-truncation). Floor 0.1. Exploration remains uniform over remaining questions.',
        uniform_cursor='Frozen at branch point; curriculum_state is authoritative for all subsequent sampling.')
    path=Path('data/bigmath-lf-curriculum.json')
    path.write_text(json.dumps(spec,indent=2)+'\n')
    print(path,sha256(path))


if __name__ == '__main__':
    main()
