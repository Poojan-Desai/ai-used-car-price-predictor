import argparse, joblib, pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--year', type=int, required=True)
p.add_argument('--miles', type=int, required=True)
p.add_argument('--make', required=True)
p.add_argument('--model', required=True)
p.add_argument('--trim', required=True)
p.add_argument('--condition', type=float, required=True)
args = p.parse_args()

pipe = joblib.load('model.pkl')
X = pd.DataFrame([{
    'year': args.year,
    'miles': args.miles,
    'make': args.make,
    'model': args.model,
    'trim': args.trim,
    'condition': args.condition
}])
print(float(pipe.predict(X)[0]))
