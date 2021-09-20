# Set options for certfile, ip, password, and toggle off
# browser auto-opening
#c.NotebookApp.certfile = u'/absolute/path/to/your/certificate/mycert.pem'
#c.NotebookApp.keyfile = u'/absolute/path/to/your/certificate/mykey.key'
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = u'*'
c.NotebookApp.password = u'sha1:68e79f6342dc:232dd9c9e1e53af7e9bf501c71272db804a70732'
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 80