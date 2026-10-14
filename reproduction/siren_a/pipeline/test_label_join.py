import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from reproduction.siren_a.pipeline.label_join import (
    audit_age_groups,
    check_identity_join,
    check_patient_overlap,
    load_metadata_index,
    load_train_targets,
    make_age_groups,
    parse_label,
)


FIELDS=["SiteID","BDSPPatientID","BidsFolder","SessionID","Age","Cognitive_Impairment"]


def make_sources(root, rows, order=None, mutate_index=None):
    root=Path(root); root.mkdir(parents=True, exist_ok=True)
    csv_path=root/"demographics.csv"; g2=root/"g2"; g2.mkdir()
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS); writer.writeheader()
        sequence=rows if order is None else [rows[i] for i in order]
        writer.writerows(sequence)
    inventory=[]; index=[]
    for i,row in enumerate(rows):
        item={"cache_name":f"cache_{i:06d}.npz","relative_path":f"{row['SiteID']}/{row['BidsFolder']}_ses-{row['SessionID']}.npz","site":row["SiteID"]}
        inventory.append(dict(item)); index.append(dict(item))
    if mutate_index: mutate_index(index)
    (g2/"inventory.json").write_text(json.dumps({"rows":inventory,"summary":{"total":len(rows)}}))
    (g2/"private_index.json").write_text(json.dumps(index))
    return csv_path,g2


class LabelJoinTests(unittest.TestCase):
    def setUp(self):
        self.rows=[
            {"SiteID":"I0002","BDSPPatientID":"p1","BidsFolder":"sub-a","SessionID":"01","Age":"55","Cognitive_Impairment":"False"},
            {"SiteID":"I0006","BDSPPatientID":"p2","BidsFolder":"sub-b","SessionID":"002","Age":"60","Cognitive_Impairment":"True"},
            {"SiteID":"S0001","BDSPPatientID":"p3","BidsFolder":"sub-c","SessionID":"3","Age":"64.9","Cognitive_Impairment":"0"},
            {"SiteID":"S0001","BDSPPatientID":"p4","BidsFolder":"sub-d","SessionID":"04","Age":"70","Cognitive_Impairment":"1"},
        ]

    def test_exact_join_and_csv_reordering_preserve_targets(self):
        with tempfile.TemporaryDirectory() as d:
            c1,g1=make_sources(Path(d)/"a",self.rows)
            c2,g2=make_sources(Path(d)/"b",self.rows,[2,0,3,1])
            a=load_metadata_index(c1,g1,expected_counts={"I0002":1,"I0006":1,"S0001":2})
            b=load_metadata_index(c2,g2,expected_counts={"I0002":1,"I0006":1,"S0001":2})
            self.assertEqual(check_identity_join(a)["matched"],4)
            ya,aa=load_train_targets(a,np.array([2,3],np.int64))
            yb,ab=load_train_targets(b,np.array([2,3],np.int64))
            np.testing.assert_array_equal(ya,yb); np.testing.assert_array_equal(aa,ab)

    def test_strict_label_spellings(self):
        for raw,want in [("True",1),("true",1),("1",1),("T",1),("Yes",1),("False",0),("false",0),("0",0),("F",0),("No",0)]:
            self.assertEqual(parse_label(raw),want)
        for raw in ("","unknown","2","none"):
            with self.assertRaises(ValueError): parse_label(raw)

    def test_duplicate_composite_key_is_rejected(self):
        rows=[dict(x) for x in self.rows]; rows[1].update({"SiteID":rows[0]["SiteID"],"BidsFolder":rows[0]["BidsFolder"],"SessionID":rows[0]["SessionID"]})
        with tempfile.TemporaryDirectory() as d:
            c,g=make_sources(d,rows)
            with self.assertRaises(ValueError): load_metadata_index(c,g,expected_counts={"I0002":2,"S0001":2})

    def test_site_bids_session_and_cache_misalignments_are_rejected(self):
        mutators=[
            lambda x:x[0].update(site="I0006"),
            lambda x:x[0].update(relative_path="I0002/wrong_ses-01.npz"),
            lambda x:x.__setitem__(slice(0,2),[x[1],x[0]]),
        ]
        for mutate in mutators:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as d:
                c,g=make_sources(d,self.rows,mutate_index=mutate)
                with self.assertRaises(ValueError): load_metadata_index(c,g,expected_counts={"I0002":1,"I0006":1,"S0001":2})

    def test_missing_or_illegal_training_label_and_age_are_rejected(self):
        for field,value in (("Cognitive_Impairment",""),("Cognitive_Impairment","maybe"),("Age",""),("Age","nan"),("Age","200")):
            rows=[dict(x) for x in self.rows]; rows[2][field]=value
            with self.subTest(field=field,value=value), tempfile.TemporaryDirectory() as d:
                c,g=make_sources(d,rows); m=load_metadata_index(c,g,expected_counts={"I0002":1,"I0006":1,"S0001":2})
                with self.assertRaises(ValueError): load_train_targets(m,np.array([2],np.int64))

    def test_holdout_and_unused_mutation_cannot_change_train_targets(self):
        with tempfile.TemporaryDirectory() as d:
            c,g=make_sources(d,self.rows); base=load_metadata_index(c,g,expected_counts={"I0002":1,"I0006":1,"S0001":2})
            y0,a0=load_train_targets(base,np.array([2,3],np.int64))
            rows=[dict(x) for x in self.rows]; rows[0]["Cognitive_Impairment"]="True"; rows[0]["Age"]="90"; rows[1]["Cognitive_Impairment"]="False"
            c2,g2=make_sources(Path(d)/"changed",rows); changed=load_metadata_index(c2,g2,expected_counts={"I0002":1,"I0006":1,"S0001":2})
            y1,a1=load_train_targets(changed,np.array([2,3],np.int64))
            np.testing.assert_array_equal(y0,y1); np.testing.assert_array_equal(a0,a1)

    def test_age_groups_boundaries_and_audit(self):
        ages=np.array([55.0,59.999,60.0,64.999,65.0,70.0])
        groups=make_age_groups(ages)
        np.testing.assert_array_equal(groups,np.array([11,11,12,12,13,14],np.int64))
        audit=audit_age_groups(groups,np.array([0,1,0,0,1,1],np.int8))
        self.assertEqual(audit["group_count"],4); self.assertEqual(audit["single_class_groups"],3)
        self.assertEqual(audit["groups_smaller_than_two"],2); self.assertEqual(audit["groups_with_both_classes"],1)

    def test_patient_namespace_overlap_and_metadata_privacy(self):
        with tempfile.TemporaryDirectory() as d:
            c,g=make_sources(d,self.rows); m=load_metadata_index(c,g,expected_counts={"I0002":1,"I0006":1,"S0001":2})
            self.assertEqual(check_patient_overlap(m,np.array([2,3]),np.array([0])),0)
            self.assertFalse(hasattr(m,"labels")); self.assertFalse(hasattr(m,"ages"))
            rows=[dict(x) for x in self.rows]; rows[0]["SiteID"]="S0001"; rows[0]["BDSPPatientID"]="p3"
            with self.assertRaises(ValueError):
                check_patient_overlap(load_metadata_index(*make_sources(Path(d)/"overlap",rows),expected_counts={"I0006":1,"S0001":3}),np.array([2,3]),np.array([0]))


if __name__=="__main__": unittest.main()
