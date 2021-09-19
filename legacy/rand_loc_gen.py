import random
import math


#  参数含义
# base_log：经度基准点，
# base_lat：维度基准点，
# radius：距离基准点的半径
def generate_random_gps(base_log=8.543611111111112, base_lat=47.37888888888889, radius=10):
    radius_in_degrees = radius / 111300
    u = float(random.uniform(0.0, 1.0))
    v = float(random.uniform(0.0, 1.0))
    w = radius_in_degrees * math.sqrt(u)
    t = 2 * math.pi * v
    x = w * math.cos(t)
    y = w * math.sin(t)
    longitude = y + base_log
    latitude = x + base_lat
    # 这里是想保留14位小数
    loga = '%.14f' % longitude
    lata = '%.14f' % latitude
    return loga, lata


longitude_, latitude_ = generate_random_gps(base_log=120.7, base_lat=30, radius=1000000)
print(longitude_)  # 经度
print(latitude_)  # 纬度

