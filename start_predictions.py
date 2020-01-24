from subprocess import check_call
import threading
import configargparse
import os

p = configargparse.ArgParser()
p.add('-d', '--base_dir', required=True, help='base directory for storing synister experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb, defaults to ``base``')
p.add('-t', required=True, help='train setup number to use for this prediction')
p.add('-p', nargs="+",required=True, help='prediction numbers')

def start_predictions(base_dir,
                      experiment,
                      train_number,
                      predict_numbers):

    scripts = [os.path.join(os.path.join(base_dir, experiment), 
                            "03_predict/setup_t{}_p{}/predict.py".format(train_number, predict_number)) for predict_number in predict_numbers]

    for script in scripts:
        thread = threading.Thread(target=check_call, args=(["python {}".format(script)]), kwargs={"shell": True})
        thread.start()

if __name__ == "__main__":
    options = p.parse_args()
    base_dir = options.base_dir
    experiment = options.e
    train_number = int(options.t)
    predict_numbers = [int(p) for p in options.p]

    start_predictions(base_dir,
                      experiment,
                      train_number,
                      predict_numbers)
