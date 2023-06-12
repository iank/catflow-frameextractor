import requests
import json

class LabelStudioAPI:
    def __init__(self, base_url, auth_token):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {auth_token}"
        }


    def import_task(self, project_id, data):
        url = f"{self.base_url}/api/projects/{project_id}/import"
        return self.make_post_request(url, data)


    def make_post_request(self, url, data):
        response = requests.post(url, headers=self.headers, data=json.dumps(data))

        if not response.ok:
            raise Exception(
                f"POST to {url} failed with status code {response.status_code}, response: {response.text}"
            )
            return None

        return response.json()


    def make_get_request(self, url):
        response = requests.get(url, headers=self.headers)

        if not response.ok:
            raise Exception(
                f"GET to {url} failed with status code {response.status_code}, response: {response.text}"
            )
            return None

        return response.json()


    def make_delete_request(self, url):
        response = requests.delete(url, headers=self.headers)

        if not response.ok:
            raise Exception(
                f"DELETE to {url} failed with status code {response.status_code}, response: {response.text}"
            )
            return None

        return True
