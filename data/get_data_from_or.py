import os
import openreview
from dotenv import load_dotenv
import ipywidgets as widgets
import json

def get_client(api_version):
    print(api_version)
    # API V2
    if api_version=="2":
        return openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=os.getenv("OR_USERNAME"),
            password=os.getenv("OR_PASSWORD"),
        )

    # API V1
    if api_version=="1":
        return openreview.Client(
            baseurl='https://api.openreview.net',
            username=os.getenv("OR_USERNAME"),
            password=os.getenv("OR_PASSWORD"),
        )
    else:
        print("Invalid API version!")
        return None
    
def main():
    load_dotenv()
    client = get_client(str(os.getenv("OR_API_VERSION")))
    print(client)
    if client is None:
        exit(1)
    notes = openreview.tools.iterget_notes(client, invitation='ICLR.cc/2019/Conference/-/Blind_Submission', details="directReplies")
    for note in notes:
        print(note)
        print(json.dumps(note.details["directReplies"], indent=3))
        break



def collapsible(dictionary):
    accordion = widgets.Accordion()
    children = []

    for key, value in dictionary.items():
        item_layout = widgets.Layout(width='auto')
        item = widgets.VBox([widgets.Text(f'{key}: {value}')], layout=item_layout)
        children.append(item)

    accordion.children = children

    for i, key in enumerate(dictionary.keys()):
        accordion.set_title(i, str(key))

    return accordion

if __name__=="__main__":
    main()