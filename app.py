import falcon, json


class Check:

    def on_post(self, req, resp):
        pass



app=falcon.API()
app.add_route("/", Check())