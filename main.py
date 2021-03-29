import os

print(os.path.isfile('./lib/saves/cnn_model.pth'))

if os.path.isfile('./lib/saves/cnn_model.pth'):
    with open("deploy.py") as f:
        code = compile(f.read(), "deploy.py", 'exec')
        exec(code)

else:
    with open("train.py") as f:
        code = compile(f.read(), "deploy.py", 'exec')
        exec(code)