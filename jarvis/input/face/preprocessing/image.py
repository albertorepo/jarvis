import cv2


class Image:
    def __init__(self, label, name, path):

        assert label is not None
        assert name is not None
        assert path is not None

        self.label = label
        self.name = name
        self.path = path

    def to_bgr(self):
        try:
            bgr = cv2.imread(self.path)
        except:
            bgr = None
        return bgr

    def to_rgb(self):
        bgr = self.to_bgr()
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = None
        return rgb

    def __repr__(self):
        return "({}, {})".format(self.label, self.name)
