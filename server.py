"""
UDP-сервер, рассылающий результат трекинга подписавшимся клиентам.

Протокол: клиент шлёт на порт сервера любую датаграмму (например b"sub") и повторяет её
хотя бы раз в секунду; клиенты, молчащие дольше CLIENT_TIMEOUT, забываются. b"bye" — отписаться.
Сервер на каждый обработанный кадр шлёт каждому клиенту одну датаграмму с JSON.
"""
import json
import socket
import threading
import time

CLIENT_TIMEOUT = 3.0


class BroadcastServer:
    def __init__(self):
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._clients: dict[tuple, float] = {}  # адрес -> когда последний раз подписывался
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._sock is not None

    @property
    def client_count(self) -> int:
        self._expire()
        with self._lock:
            return len(self._clients)

    def start(self, host: str, port: int):
        """Бросает OSError, если порт занят и т.п."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((host, port))
        except OSError:
            sock.close()
            raise
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            # Windows: иначе ICMP «порт недоступен» от ушедшего клиента ломает recvfrom
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._recv_loop, args=(sock,), daemon=True)
        self._thread.start()

    def stop(self):
        sock, self._sock = self._sock, None
        if sock is None:
            return
        sock.close()
        if self._thread:
            self._thread.join(timeout=2)
        with self._lock:
            self._clients.clear()

    def _recv_loop(self, sock: socket.socket):
        while self._sock is sock:
            try:
                msg, addr = sock.recvfrom(256)
            except socket.timeout:
                continue
            except ConnectionError:
                continue
            except OSError:
                break
            with self._lock:
                if msg.strip() == b"bye":
                    self._clients.pop(addr, None)
                else:
                    self._clients[addr] = time.monotonic()

    def _expire(self):
        now = time.monotonic()
        with self._lock:
            for addr in [a for a, t in self._clients.items() if now - t > CLIENT_TIMEOUT]:
                del self._clients[addr]

    def broadcast(self, data: dict):
        sock = self._sock
        if sock is None:
            return
        self._expire()
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        packet = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
        for addr in clients:
            try:
                sock.sendto(packet, addr)
            except OSError:
                pass  # UDP: недоставленное просто теряется
