import socket
sock = socket.socket()
sock.bind(('localhost', 0))
host, port = sock.getsockname()
socket.accept()
