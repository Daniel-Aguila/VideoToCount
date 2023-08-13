import cgi

form = cgi.FieldStorage()
searchVideo = form.getvalue('searchbox')

print(searchVideo)