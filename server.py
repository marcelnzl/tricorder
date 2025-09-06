import http.server
import ssl
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# The directory you want to serve files from
web_dir = os.path.join(os.path.dirname(__file__), 'src')
os.chdir(web_dir)

# Server details from environment variables
host = os.getenv("HOST_IP", "192.168.1.35")  # Listen on all available network interfaces
port = int(os.getenv("FRONTEND_PORT", "8081"))
ssl_key_path = os.getenv("SSL_KEY_PATH", "key.pem")
ssl_cert_path = os.getenv("SSL_CERT_PATH", "cert.pem")

# Create the server
httpd = http.server.HTTPServer(
    (host, port), 
    http.server.SimpleHTTPRequestHandler
)

# Wrap the server socket with SSL/TLS
httpd.socket = ssl.wrap_socket(
    httpd.socket,
    keyfile=f"../{ssl_key_path}",    # Path to your key file
    certfile=f"../{ssl_cert_path}",  # Path to your certificate file
    server_side=True
)

print(f"Serving HTTPS on https://{host}:{port} from directory {web_dir}...")
httpd.serve_forever()
