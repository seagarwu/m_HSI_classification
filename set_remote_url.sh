#!/bin/sh
git branch -M main
git remote rm github
git remote add github git@github.com:seagarwu/m_HSI_classification.git
git remote -v
