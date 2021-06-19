import json

def get_image_json(json_=None, img_name=None):
    
    temp_dict = {}
    _found = False
    for img in json_["images"]:
        if img["file_name"] == img_name:
            temp_dict = img
            _found = True
            break
    
    if _found == False:        
        #insert a new element
        img = json_["images"][-1]
        temp_dict["id"] = int(img["id"]) + 1
        temp_dict["license"] = img["license"]
        temp_dict["file_name"] = img_name
        temp_dict["height"] = img["height"]
        temp_dict["width"] = img["width"]
        temp_dict["date_captured"] = img["date_captured"]
        
        json_["images"].append(temp_dict)
        
    return json_, temp_dict


def insert_bbox(json_file, coordinates, img_id, category_id):
    temp_dict = {}
    temp_dict["id"] = (json_file["annotations"][-1]["id"] + 1)
    temp_dict["image_id"] = img_id
    temp_dict["category_id"] = category_id
    temp_dict["bbox"] = coordinates
    temp_dict["area"] = coordinates[2] * coordinates[3]
    temp_dict["segmentation"] = []
    temp_dict["iscrowd"] = 0
    
    #append to the json
    json_file["annotations"].append(temp_dict)
    return json_file
    