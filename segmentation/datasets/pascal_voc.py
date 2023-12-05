# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

from mmseg.datasets import DATASETS
from mmseg.datasets import PascalVOCDataset as _PascalVOCDataset


@DATASETS.register_module(force=True)
class PascalVOCDataset(_PascalVOCDataset):

    CLASSES = (
        'background',
        'aeroplane',
               'bicycle',
               'bird',
               'boat',
               'bottle',
               'bus',
               'car',
               'cat',
               'chair',
               'cow',
               'table',
               'dog',
               'horse',
               'motorbike',
               'person',
               'plant',
               'sheep',
               'sofa',
               'train',
               'monitor')
    def __init__(self,**kwargs):
        super(PascalVOCDataset, self).__init__(reduce_zero_label=False,**kwargs)
    # CLASSES = ('background',
    #            'airplane or aeroplane (informally plane), is a fixed-wing aircraft that is propelled forward by thrust from a jet engine, propeller, or rocket engine',
    #            'bicycle, also called a pedal cycle, bike or cycle, is a human-powered or motor-powered assisted, pedal-driven, single-track vehicle, having two wheels attached to a frame, one behind the other',
    #            'bird, is a group of warm-blooded vertebrates constituting the class Aves (), characterised by feathers, toothless beaked jaws, the laying of hard-shelled eggs, a high metabolic rate, a four-chambered heart, and a strong yet lightweight skeleton',
    #            'boat, is a watercraft of a large range of types and sizes, but generally smaller than a ship, which is distinguished by its larger size, shape, cargo or passenger capacity, or its ability to carry boats',
    #
    #            'bottle, is a narrow-necked container made of an impermeable material (such as glass, plastic or aluminium) in various shapes and sizes that stores and transports liquids',
    #            'bus (contracted from omnibus, with variants multibus, motorbus, autobus, etc.), is a road vehicle that carries significantly more passengers than an average car or van',
    #            'car (or automobile), is a wheeled motor vehicle that is used for transportation',
    #            'cat (Felis catus), is a domestic species of small carnivorous mammal',
    #            'chair, is a type of seat, typically designed for one person and consisting of one or more legs, a flat seat and a back-rest',
    #            'cow (Cattle), is large, domesticated, cloven-hooved, herbivores',
    #            'table, is an item of furniture with a raised flat top and is supported most commonly by 1 or 4 legs (although some can have more), used as a surface for working at, eating from or on which to place things',
    #            'dog (Canis familiaris or Canis lupus familiaris), is a domesticated descendant of the wolf',
    #            'horse (Equus ferus caballus), is a domesticated, one-toed, hoofed mammal',
    #            'motorcycle, often called a motorbike, bike, cycle, or (if three-wheeled) trike, is a two- or three-wheeled motor vehicle',
    #            'person (PL: people or persons), is a being that has certain capacities or attributes such as reason, morality, consciousness or self-consciousness, and being a part of a culturally established form of social relations such as kinship, ownership of property, or legal responsibility',
    #            'plants, are predominantly photosynthetic eukaryotes of the kingdom Plantae',
    #            'sheep or domestic sheep (Ovis aries), are domesticated, ruminant mammals typically kept as livestock',
    #            'couch, also known as a sofa, settee, or chesterfield, is a cushioned item of furniture for seating multiple people (although it is not uncommon for a single person to use a couch)',
    #            'train (from Old French trahiner, from Latin trahere, \"to pull, to draw\"), is a series of connected vehicles that run along a railway track and transport people or freight In rail transport.',
    #            'monitor, is an output device that displays information in pictorial or textual form')

    # CLASSES = ('background',
    #            'airplane or aeroplane (informally plane)',
    #            'bicycle, also called a pedal cycle, bike or cycle',
    #            'bird',
    #            'boat, is a watercraft (ship)',
    #
    #            'bottle',
    #            'bus',
    #            'car',
    #            'cat',
    #            'chair',
    #            'cow',
    #            'table',
    #            'dog',
    #            'horse',
    #            'motorbike, also called motorcycle',
    #            'person',
    #            'plant',
    #            'sheep',
    #            'sofa',
    #            'train',
    #            'monitor')