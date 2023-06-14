import requests
import json


class LabelStudioAPI:
    """
    A class to interact with the Label Studio API.

    Args:
        base_url (str): The base URL for the Label Studio API.
        auth_token (str): The authentication token for the Label Studio API.
    """

    def __init__(self, base_url, auth_token):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {auth_token}",
        }

    def import_task(self, project_id, data):
        """
        Import a task into a project.

        Args:
            project_id (int): The ID of the project.
            data (dict): The task data to import.

        Returns:
            dict: The response from the API.

        Raises:
            HTTPError: If the request to the Label Studio API fails.
        """
        url = f"{self.base_url}/api/projects/{project_id}/import"
        return self.make_post_request(url, data)

    def create_snapshot(self, project_id):
        """
        Create an export snapshot of a project.

        Args:
            project_id (int): The ID of the project.

        Returns:
            tuple: A tuple containing the snapshot ID and title.

        Raises:
            HTTPError: If the request to the Label Studio API fails.
        """
        url = f"{self.base_url}/api/projects/{project_id}/exports"
        response = self.make_post_request(
            url, {"task_filter_options": {"only_with_annotations": True}}
        )
        return (response.get("id"), response.get("title")) if response else None

    def check_export_status(self, project_id, export_id):
        """
        Check the status of an export snapshot.

        Args:
            project_id (int): The ID of the project.
            export_id (int): The ID of the snapshot.

        Returns:
            bool: True if the export has completed, False otherwise.

        Raises:
            ValueError: If the export fails or is not found.
            HTTPError: If the request to the Label Studio API fails.
        """
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
        """
        Check the status of a snapshot conversion.

        Args:
            project_id (int): The ID of the project.
            export_id (int): The ID of the snapshot.
            format_type (str): The format to check.

        Returns:
            bool: True if the conversion has completed, False otherwise.

        Raises:
            ValueError: If the conversion failed or is not found.
            HTTPError: If the request to the Label Studio API fails.
        """
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
        """
        Convert a snapshot into a different format.

        Args:
            project_id (int): The ID of the project.
            export_id (int): The ID of the snapshot.
            format_type (str): The format to convert to.

        Raises:
            HTTPError: If the request to the Label Studio API fails.
        """
        url = f"{self.base_url}/api/projects/{project_id}/exports/{export_id}/convert"
        self.make_post_request(url, {"export_type": format_type})

    def download_snapshot(self, project_id, export_id):
        """
        Download a snapshot from a project.

        Args:
            project_id (int): The ID of the project.
            export_id (int): The ID of the snapshot.

        Returns:
            requests.models.Response: The HTTP response object.
        """
        url = f"{self.base_url}/api/projects/{project_id}/exports/{export_id}/download?exportType=YOLO"
        req = requests.get(url, headers=self.headers, stream=True)

        return req

    def make_post_request(self, url, data):
        """
        Make a POST request to the Label Studio API.

        Args:
            url (str): The URL to make the request to.
            data (dict): The data to send in the request.

        Returns:
            dict: The response from the API.

        Raises:
            HTTPError: If the response status code is not OK.
        """
        response = requests.post(url, headers=self.headers, data=json.dumps(data))
        response.raise_for_status()

        return response.json()

    def make_get_request(self, url):
        """
        Make a GET request to the Label Studio API.

        Args:
            url (str): The URL to make the request to.

        Returns:
            dict: The response from the API.

        Raises:
            HTTPError: If the response status code is not OK.
        """
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def make_delete_request(self, url):
        """
        Make a DELETE request to the Label Studio API.

        Args:
            url (str): The URL to make the request to.

        Returns:
            bool: True if the request was successful, False otherwise.

        Raises:
            HTTPError: If the response status code is not 200.
        """
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()

        return True
