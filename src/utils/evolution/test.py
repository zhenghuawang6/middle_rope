# import socket
# sock = socket.socket()
# sock.bind(('localhost', 0))
# host, port = sock.getsockname()
# print(host,port)
# sock.listen(1)
# conn, addr = sock.accept()
# print("连接成功")

import json
a = {"data":[21.35,34]}
print(json.dumps(a))
