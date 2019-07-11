import uuid


class Config(object):
    UPLOAD_FOLDER = './uploads'
    TEST_FOLDER = './test_images'
    WEIGHTS_FOLDER = './weights_img'
    SECRET_KEY = uuid.uuid4().hex
    DEBUG = False
    TESTING = False
