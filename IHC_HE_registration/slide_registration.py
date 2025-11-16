
"""
(C) This code belongs to https://github.com/bifold-pathomics/xMIL-Pathways
Please see the citation and copyright instructions in the above-mentioned repository.

For this script, you should install VALIS (https://valis.readthedocs.io/en/latest/index.html)
"""

import os
import json
import os.path as op
import argparse

import valis.registration as registration_valis


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--patient_id', type=str, required=True)
    parser.add_argument('--patient_slides-dir', type=str, required=True)
    parser.add_argument('--crop', type=str, default=None)
    parser.add_argument('--do-not-save', action='store_true')
    parser.add_argument('--micro-registration', action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    slides_this_patient = [p for p in os.listdir(args.patient_slides_dir) if
                   p.endswith('.svs') or p.endswith('.tiff') and not p.startswith('.')]

    print('-------------------------')
    print('patient ID: ', args.patient_id)
    print('slides of this patient: ~~~~')
    print(slides_this_patient)

    # patient slide info  -------------------
    patient_HnE = [op.join(args.slide_src_dir, name) for name in slides_this_patient if 'HE' in name][0]
    print('HnE: ', patient_HnE)
    print('-------------------------')

    slides_this_patient = [op.join(args.slide_src_dir, s) for s in slides_this_patient]

    # save folder -------------------
    results_dst_dir = os.path.join(args.results_dir, args.patient_id)
    os.makedirs(results_dst_dir, exist_ok=True)
    registered_slide_dst_dir = op.join(results_dst_dir, 'registered_slides')
    os.makedirs(registered_slide_dst_dir, exist_ok=True)

    print(f"Results will be written to: {results_dst_dir}")
    with open(os.path.join(results_dst_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # registrar info -------------------
    registrar = registration_valis.Valis(args.slide_src_dir, results_dst_dir, align_to_reference=True,
                                         reference_img_f=patient_HnE, crop=args.crop,
                                         img_list=slides_this_patient)

    _, _, _ = registrar.register()
    if args.micro_registration:
        registrar.register_micro(max_non_rigid_registration_dim_px=2000, align_to_reference=True)

    # Save all registered slides -------------------
    if not args.do_not_save:
        print('warping and saving .....')
        registrar.warp_and_save_slides(registered_slide_dst_dir, compression='jpeg', Q=90)

    registration_valis.kill_jvm()


if __name__ == '__main__':
    main()
