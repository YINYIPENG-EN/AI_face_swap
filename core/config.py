import insightface
import core.globals

FACE_ANALYSER = None


def get_face_analyser():
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        # FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=core.globals.providers)
        print('core.globals.providers:', core.globals.providers)
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', root='F:/roop/',providers=core.globals.providers)

        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_face(img_data):
    face = get_face_analyser().get(img_data)
    try:
        return sorted(face, key=lambda x: x.bbox[0])[0]
    except IndexError:
        return None
