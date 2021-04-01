import os

test_txt = "testet/test/test.jpg"
txt = test_txt.split(os.sep)[-1]

print(txt)