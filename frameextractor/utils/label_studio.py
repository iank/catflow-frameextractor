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


    def create_snapshot(self, project_id):
        url = f"{self.base_url}/api/projects/{project_id}/exports"
        response = self.make_post_request(
            url, {"task_filter_options": {"only_with_annotations": True}}
        )
        return (response.get("id"), response.get("title")) if response else None


    def check_export_status(self, project_id, export_id):
        url = f"{self.base_url}/api/projects/{project_id}/exports"
        response = self.make_get_request(url)

        for export in response:
            if export["id"] == export_id:
                if export["status"] == "completed":
                    return True
                elif export["status"] == "failed":
                    raise ValueError(f"Export failed with ID: {export_id}")
        raise ValueError(f"No export found with ID: {export_id}")


    def check_conversion_status(self, project_id, export_id, format_type):
        url = f"{self.base_url}/api/projects/{project_id}/exports"
        response = self.make_get_request(url)

        for export in response:
            if export["id"] == export_id:
                for format in export["converted_formats"]:
                    if format["export_type"] == format_type:
                        if format["status"] == "completed":
                            return True
                        elif format["status"] == "failed":
                            raise ValueError(
                                f"Conversion failed for format {format_type} with export ID: {export_id}"
                            )
        raise ValueError(
            f"No export found with ID: {export_id} or format: {format_type}"
        ) 


    def convert_snapshot(self, project_id, export_id, format_type):
        url = f"{self.base_url}/api/projects/{project_id}/exports/{export_id}/convert"
        self.make_post_request(url, {"export_type": format_type})


    def download_snapshot(self, project_id, export_id, snapshot_name):
        url = f"{self.base_url}/api/projects/{project_id}/exports/{export_id}/download?exportType=YOLO"
        req = requests.get(url, headers=self.headers, stream=True)

        return req


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
