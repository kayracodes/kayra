import http.server
import socketserver

PORT = 8000
# Handler = http.server.SimpleHTTPRequestHandler

# with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
#     print(f"Serving at port {PORT}")
#     httpd.serve_forever()


class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.path = "/wounded.html"
        return http.server.SimpleHTTPRequestHandler.do_GET(self)


Handler = MyRequestHandler
with socketserver.TCPServer(("192.168.79.147", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
