from datetime import datetime,timedelta
import json

base_folder = '../experiments/scenario-9-sbrt-timed-08M'

log_filenames = [
    base_folder + '/baseline/test.log',
    base_folder + '/by-epoch/DL-CD-MAP-GPU-EP.log',
    base_folder + '/by-topology-4-layers/DL-CD-MAP-GPU-4L.log',
    base_folder + '/by-topology-5-layers/DL-CD-MAP-GPU-5L.log',
    base_folder + '/by-topology-6-layers/DL-CD-MAP-GPU-6L.log',
]

code = ''
optimizer = ''
nb_epoch = ''
fit_time = ''
pred_time_start = ''
pred_time_end = ''
pred_time = ''
bers = []
ans = {}

for filename in log_filenames:
    
    with open(filename) as in_file:
        for line in in_file:
            parts = line.replace('\t',' ').split(' ')
            if 'fit started' in line:
                code = parts[2]
                nb_epoch = parts[6].split('=')[1]
                optimizer = parts[4]
                ans.setdefault(code, {})
                ans[code].setdefault(nb_epoch, {})
                ans[code][nb_epoch].setdefault(optimizer, {})
            elif 'fit finished' in line:
                fit_time = timedelta(seconds=float(parts[10]))
                ans[code][nb_epoch][optimizer]['fit_time'] = fit_time
            elif 'test @ sigmas(dB)' in line:
                pred_time_start = datetime.strptime(parts[0] + ' ' + parts[1], '%Y-%m-%d %H:%M:%S.%f')
            elif 'test @ sigma[10]' in line:
                pred_time_end = datetime.strptime(parts[0] + ' ' + parts[1], '%Y-%m-%d %H:%M:%S.%f')
                pred_time = pred_time_end - pred_time_start
                ans[code][nb_epoch][optimizer]['pred_time'] = pred_time
            elif 'nb_bits' in line:
                sigma_db = float(parts[5].split('=')[1])
                nb_bits = float(parts[6].split('=')[1])
                nb_errors = float(parts[7].split('=')[1])
                ans[code][nb_epoch][optimizer].setdefault('x', [])
                ans[code][nb_epoch][optimizer]['x'].append(sigma_db)
                ans[code][nb_epoch][optimizer].setdefault('y', [])
                ans[code][nb_epoch][optimizer]['y'].append(nb_errors/nb_bits)

    with open(filename.replace('.log','.json'), 'w') as out_file:
        json.dump(ans, out_file, indent=4, sort_keys=True, default=str)