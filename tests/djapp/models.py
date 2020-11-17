# Copyright: See the LICENSE file.

"""Helpers for testing django apps."""

import os.path

from django.conf import settings
from django.db import models

try:
    from PIL import Image
except ImportError:
    Image = None

import django
from django.conf import settings
from django.db import models

class StandardModel(models.Model):
    foo = models.CharField(max_length=20)


class NonIntegerPk(models.Model):
    foo = models.CharField(max_length=20, primary_key=True)
    bar = models.CharField(max_length=20, blank=True)


class MultifieldModel(models.Model):
    slug = models.SlugField(max_length=20, unique=True)
    text = models.CharField(max_length=20)


class MultifieldUniqueModel(models.Model):
    slug = models.SlugField(max_length=20, unique=True)
    text = models.CharField(max_length=20, unique=True)
    title = models.CharField(max_length=20, unique=True)


class AbstractBase(models.Model):
    foo = models.CharField(max_length=20)

    class Meta:
        abstract = True


class ConcreteSon(AbstractBase):
    pass


class AbstractSon(AbstractBase):
    class Meta:
        abstract = True


class ConcreteGrandSon(AbstractSon):
    pass


class StandardSon(StandardModel):
    pass


class PointedModel(models.Model):
    foo = models.CharField(max_length=20)


class PointerModel(models.Model):
    bar = models.CharField(max_length=20)
    pointed = models.OneToOneField(
        PointedModel,
        related_name='pointer',
        null=True,
        on_delete=models.CASCADE
    )


class WithDefaultValue(models.Model):
    foo = models.CharField(max_length=20, default='')


WITHFILE_UPLOAD_TO = 'django'
WITHFILE_UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, WITHFILE_UPLOAD_TO)


class WithFile(models.Model):
    afile = models.FileField(upload_to=WITHFILE_UPLOAD_TO)


if Image is not None:  # PIL is available

    class WithImage(models.Model):
        animage = models.ImageField(upload_to=WITHFILE_UPLOAD_TO)
        size = models.IntegerField(default=0)

else:
    class WithImage(models.Model):
        pass


class WithSignals(models.Model):
    foo = models.CharField(max_length=20)


class CustomManager(models.Manager):

    def create(self, arg=None, **kwargs):
        return super().create(**kwargs)


class WithCustomManager(models.Model):

    foo = models.CharField(max_length=20)

    objects = CustomManager()


class AbstractWithCustomManager(models.Model):
    custom_objects = CustomManager()

    class Meta:
        abstract = True


class FromAbstractWithCustomManager(AbstractWithCustomManager):
    pass


# For auto_fields
# ===============


class MultiFieldModel(models.Model):
    # Text
    char = models.CharField(max_length=4)  # Below FuzzyText' boundary
    text = models.TextField()
    slug = models.SlugField()

    # Misc
    binary = models.BinaryField()
    boolean = models.BooleanField(default=False)
    nullboolean = models.NullBooleanField()
    uuid = models.UUIDField()

    # Date and time
    date = models.DateField()
    datetime = models.DateTimeField()
    datetime_auto_now = models.DateTimeField(auto_now=True)
    datetime_auto_now_add = models.DateTimeField(auto_now_add=True)
    time = models.TimeField()
    duration = models.DurationField()

    # Numbers
    int = models.IntegerField()
    dec = models.DecimalField(max_digits=10, decimal_places=4)
    bigint = models.BigIntegerField()
    posint = models.PositiveIntegerField()
    smallint = models.SmallIntegerField()
    smallposint = models.PositiveSmallIntegerField()
    float = models.FloatField()

    # Filed
    attached = models.FileField()
    img = models.ImageField()

    # Internet
    ipv4 = models.GenericIPAddressField(protocol='ipv4')
    ipv6 = models.GenericIPAddressField(protocol='ipv6')
    ipany = models.GenericIPAddressField()
    email = models.EmailField()
    url = models.URLField()


class OptionalModel(models.Model):
    CHAR_LEN = 19

    char_req = models.CharField(max_length=CHAR_LEN)
    char_blank = models.CharField(max_length=CHAR_LEN, blank=True)
    char_null = models.CharField(max_length=CHAR_LEN, null=True)
    char_blank_null = models.CharField(max_length=CHAR_LEN, blank=True, null=True)
    char_blank_null_default = models.CharField(max_length=CHAR_LEN, blank=True, null=True, default='hello world')

    int_req = models.IntegerField()
    int_blank = models.IntegerField(blank=True)
    int_null = models.IntegerField(null=True)
    int_blank_null = models.IntegerField(blank=True, null=True)
    int_blank_null_default = models.IntegerField(blank=True, null=True, default=5)


class ForeignKeyModel(models.Model):
    name = models.CharField(max_length=20)
    target = models.ForeignKey(ComprehensiveMultiFieldModel, on_delete=models.CASCADE)


class OneToOneModel(models.Model):
    name = models.CharField(max_length=20)
    relates_to_req = models.OneToOneField(ForeignKeyModel, on_delete=models.CASCADE)
    relates_to_opt = models.OneToOneField(OptionalModel, on_delete=models.CASCADE, blank=True, null=True)


class ManyToManySourceModel(models.Model):
    name = models.CharField(max_length=20)
    targets_req = models.ManyToManyField(ComprehensiveMultiFieldModel, blank=False)
    targets_opt = models.ManyToManyField(OptionalModel, blank=True)


class ManyToManyThroughModel(models.Model):
    name = models.CharField(max_length=20)
    multi = models.ForeignKey(ComprehensiveMultiFieldModel, on_delete=models.CASCADE)
    source = models.ForeignKey('ManyToManyWithThroughSourceModel', on_delete=models.CASCADE)


class ManyToManyWithThroughSourceModel(models.Model):
    name = models.CharField(max_length=20)
    targets = models.ManyToManyField(MultiFieldModel, through=ManyToManyThroughModel)


class CycleAModel(models.Model):
    a_name = models.CharField(max_length=10)
    c_fkey = models.ForeignKey('CycleCModel', on_delete=models.CASCADE, null=True)


class CycleBModel(models.Model):
    b_name = models.CharField(max_length=10)
    a_fkey = models.ForeignKey(CycleAModel, on_delete=models.CASCADE)


class CycleCModel(models.Model):
    c_name = models.CharField(max_length=10)
    b_fkey = models.ForeignKey(CycleBModel, on_delete=models.CASCADE)


class OrderWithRespectTo(models.Model):
    fk = models.ForeignKey(ComprehensiveMultiFieldModel, on_delete=models.CASCADE)

    class Meta:
        order_with_respect_to = 'fk'


class ParentModel(models.Model):
    char_req = models.CharField(max_length=20)
    char_opt = models.CharField(max_length=20, blank=True, null=True)

    int_req = models.IntegerField()
    int_opt = models.IntegerField(blank=True, null=True)

    fk = models.ForeignKey(ComprehensiveMultiFieldModel, on_delete=models.CASCADE)

    class Meta:
        abstract = True


class ChildModel(ParentModel):
    char_req2 = models.CharField(max_length=20)
    char_opt2 = models.CharField(max_length=20, blank=True, null=True)

    int_req2 = models.IntegerField()
    int_opt2 = models.IntegerField(blank=True, null=True)

    fk2 = models.ForeignKey(OptionalModel, on_delete=models.CASCADE)


class MultiTableParentModel(models.Model):
    char_req = models.CharField(max_length=20)
    char_opt = models.CharField(max_length=20, blank=True, null=True)

    int_req = models.IntegerField()
    int_opt = models.IntegerField(blank=True, null=True)

    fk = models.ForeignKey(ComprehensiveMultiFieldModel, on_delete=models.CASCADE)


class MultiTableChildModel(MultiTableParentModel):
    char_req2 = models.CharField(max_length=20)
    char_opt2 = models.CharField(max_length=20, blank=True, null=True)

    int_req2 = models.IntegerField()
    int_opt2 = models.IntegerField(blank=True, null=True)

    fk2 = models.ForeignKey(OptionalModel, on_delete=models.CASCADE)