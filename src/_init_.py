#!/usr/bin/python
# -*- coding: utf-8 -*-

import ConfigParser

def ConfigSectionMap(section="data"):
    dict_ = {}
    options = Parser.options(section)
    for option in options:
        try:
            dict_[option] = Parser.get(section, option)
            if dict_[option] == -1:
                print "Skipped: %s" % option
        except:
            print "Exception on %s!" % option
            dict_[option] = None
    return dict_

Parser = ConfigParser.ConfigParser()
Parser.read("../common.config")
config_data = ConfigSectionMap("data")
config_log = ConfigSectionMap("log")