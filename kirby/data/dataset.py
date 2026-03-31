import os
import logging
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import datetime
import copy
from skimage.transform import resize

import msgpack
import h5py
import torch
import itertools
import numpy as np
import torch.nn.functional as F

from kirby.data import Data, Interval

@dataclass
class DatasetIndex:
    """Accessing the dataset is done by specifying a session id and a time interval."""

    session_id: str
    start: float
    end: float


class Dataset(torch.utils.data.Dataset):
    r"""This class abstracts a collection of lazily-loaded Data objects. Each of these
    Data objects corresponds to a session and lives on the disk until it is requested.
    The `include` argument guides which sessions are included in this Dataset.
    To request a piece of a included session's data, you can use the `get` method,
    or index the Dataset with a `DatasetIndex` object (see `__getitem__`).

    This definition is a deviation from the standard PyTorch Dataset definition, which
    generally presents the dataset directly as samples. In this case, the Dataset
    by itself does not provide you with samples, but rather the means to flexibly work
    and accesss complete sessions.
    Within this framework, it is the job of the sampler to provide the
    DatasetIndex indices to slice the dataset into samples (see `kirby.data.sampler`).

    Files will be opened, and only closed when the Dataset object is deleted.

    Args:
        root: The root directory of the dataset.
        split: The split of the dataset. This is used to determine the sampling intervals
            for each session.
        include: A list of dictionaries specifying the datasets to include. Each dictionary
            should have the following keys:
            - dandiset: The dandiset to include.
            - selection: A dictionary specifying the selection criteria for the dataset.
        transform: A transform to apply to the data. This transform should be a callable
            that takes a Data object and returns a Data object.
    """

    _check_for_data_leakage_flag: bool = True
    _open_files: Optional[Dict[str, h5py.File]] = None
    _data_objects: Optional[Dict[str, Data]] = None

    def __init__(
        self,
        root: str,
        split: str,
        include: List[Dict[str, Any]],
        transform=None,
        pretrain=False,
        finetune=False,
        small_model=False,
        task='movie_decoding',
        model_dim=512,
        zscore_normalize=False,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.pretrain = pretrain
        self.finetune = finetune
        self.small_model = small_model # for scaling
        self.task = task
        self.zscore_normalize = zscore_normalize
        self.model_dim = model_dim
        if self.model_dim <= 512 and 'movie_decoding' in self.task:
            H = 64
            W = 128
        else:
            H = 128
            W = 256

        print('task is', self.task)


        def resize_frames_uint8(arr, Ht, Wt):
            """
            arr: (T, H, W)  # grayscale, integer type (uint8 recommended)
            size_hw: (H_out, W_out)  # target height, width
            returns: (T, H_out, W_out), dtype=uint8
            """
            T, H, W = arr.shape
            size_wh = (Wt, Ht)  # PIL expects (W, H)

            # clip and cast to uint8 if needed
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)

            out = np.empty((T, Ht, Wt), dtype=np.uint8)
            for i in range(T):
                im = Image.fromarray(arr[i], mode='L')
                # LANCZOS provides good anti-aliasing for large downscales
                im_r = im.resize(size_wh, Image.Resampling.LANCZOS)
                out[i] = np.asarray(im_r, dtype=np.uint8)
            return out

        if self.task == 'movie_decoding_one':
            from allensdk.core.brain_observatory_cache import BrainObservatoryCache
            from PIL import Image
            boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
            data_set = boc.get_ophys_experiment_data(501940850)
            movie = data_set.get_stimulus_template('natural_movie_one') # (900, 304, 608)
            self.movie_frames = resize_frames_uint8(movie, H, W)
        elif self.task == 'movie_decoding_three':
            from allensdk.core.brain_observatory_cache import BrainObservatoryCache
            from PIL import Image
            boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
            data_set = boc.get_ophys_experiment_data(501940850)
            movie = data_set.get_stimulus_template('natural_movie_three') # (3600, 304, 608)
            self.movie_frames = resize_frames_uint8(movie, H, W)
            
        elif self.task == 'scene_decoding':
            from allensdk.core.brain_observatory_cache import BrainObservatoryCache
            from PIL import Image
            boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
            data_set = boc.get_ophys_experiment_data(503772253)
            scene = data_set.get_stimulus_template('natural_scenes') # (118, 918, 1174)
            self.movie_frames = resize_frames_uint8(scene, H, W)


        if include is None:
            raise ValueError("Please specify the datasets to include")

        self.include = include
        self.transform = transform

        self.session_info_dict, self.session_ids, self.unit_ids = self._look_for_files()

        self._open_files = {
            session_id: h5py.File(session_info["filename"], "r")
            for session_id, session_info in self.session_info_dict.items()
        }

        self._data_objects = {
            session_id: Data.from_hdf5(f, lazy=True)
            for session_id, f in self._open_files.items()
        }

        # Session-level z-score normalization for calcium traces
        self._zscore_stats = {}
        if self.zscore_normalize:
            for session_id, data in self._data_objects.items():
                df_f = data.calcium_traces.df_over_f
                mean = float(np.mean(df_f))
                std = float(np.std(df_f))
                if std == 0:
                    std = 1.0
                self._zscore_stats[session_id] = (mean, std)
                print(f"[z-score] {session_id}: mean={mean:.4f}, std={std:.4f}")

        if self.task == 'movie_decoding_one':
            for sid, dobj in self._data_objects.items():
                valid_mask = (dobj.natural_movie_one.end - dobj.natural_movie_one.start) != 0.0
                dobj.natural_movie_one.train_mask = dobj.natural_movie_one.train_mask & valid_mask
                dobj.natural_movie_one.valid_mask = dobj.natural_movie_one.valid_mask & valid_mask
                dobj.natural_movie_one.test_mask  = dobj.natural_movie_one.test_mask  & valid_mask
        
        elif self.task == 'movie_decoding_three':
            for sid, dobj in self._data_objects.items():
                valid_mask = (dobj.natural_movie_three.end - dobj.natural_movie_three.start) != 0.0
                dobj.natural_movie_three.train_mask = dobj.natural_movie_three.train_mask & valid_mask
                dobj.natural_movie_three.valid_mask = dobj.natural_movie_three.valid_mask & valid_mask
                dobj.natural_movie_three.test_mask  = dobj.natural_movie_three.test_mask  & valid_mask
        
        elif self.task == 'scene_decoding':
            for sid, dobj in self._data_objects.items():
                valid_mask = (dobj.natural_scenes_trials.end - dobj.natural_scenes_trials.start) != 0.0
                dobj.natural_scenes_trials.train_mask = dobj.natural_scenes_trials.train_mask & valid_mask
                dobj.natural_scenes_trials.valid_mask = dobj.natural_scenes_trials.valid_mask & valid_mask
                dobj.natural_scenes_trials.test_mask  = dobj.natural_scenes_trials.test_mask  & valid_mask
        
        whole = len(self._data_objects)

        self.valid_session_ids = [session_id for session_id in self._data_objects.keys()
                                    if self._is_valid_sample(self._data_objects[session_id].subject.cre_line, session_id)
                            ]
        
        self._data_objects = {session_id: self._data_objects[session_id] for session_id in self.valid_session_ids}

        # Filter session_info_dict to only include valid session IDs
        self.session_info_dict = {
            key: value for key, value in self.session_info_dict.items() if key in self.valid_session_ids
        }

        # Filter session_ids to only include valid session IDs
        self.session_ids = [session_id for session_id in self.session_ids if session_id in self.valid_session_ids]

        # Filter unit_ids to only include units belonging to valid sessions
        self.unit_ids = [
            unit_id for unit_id in self.unit_ids if (unit_id.split('/')[0] + '/' + unit_id.split('/')[1]) in self.valid_session_ids
        ]

        print(f"Filtered session_info_dict: {len(self.session_info_dict)}")
        print(f"Filtered session_ids: {len(self.session_ids)}")
        print(f"Filtered unit_ids: {len(self.unit_ids)}")
        print(f"Filtered dataset size: {len(self.valid_session_ids)} / {whole}")

    '''
    def _generate_all_sample_indices(self):
        indices = []
        for session_id, dobj in self._data_objects.items():
            movie = dobj.natural_movie_one
            for start, end in zip(movie.start, movie.end):
                index = DatasetIndex(session_id=session_id, start=float(start), end=float(end))
                indices.append(index)
        return indices
    
    def _is_valid(self, index: DatasetIndex):
        return not np.isclose(index.end, index.start)
    '''

    # CRE lines used for pretraining (statistically regular, predictable neurons).
    # Pretrain uses these four lines; finetune uses the remaining (unpredictable) lines.
    PREDICTABLE_CRE_LINES = ['SST_IRES_CRE', 'VIP_IRES_CRE', 'PVALB_IRES_CRE', 'NTSR1_CRE_GN220']

    def _is_valid_sample(self, cre_line, session_id):
        """Check whether a session should be included based on the CRE-line curriculum.

        Pretrain: restricted to predictable (regular-firing) CRE lines.
        Finetune: restricted to unpredictable (stimulus-driven) CRE lines.
        """
        # Pretrain on predictable CRE lines only (4 lines: SST, VIP, PVALB, NTSR1).
        if self.pretrain and cre_line not in self.PREDICTABLE_CRE_LINES:
            return False
        # Finetune on unpredictable CRE lines only (remaining 9 lines).
        if self.finetune and cre_line in self.PREDICTABLE_CRE_LINES:
            return False
        else:
            if self.small_model:
                if int(session_id.split('/')[1]) % 10 != 0:
                    return False
        return True

    def _close_open_files(self):
        """Closes the open files and deletes open data objects.
        This is useful when you are done with the dataset.
        """
        if self._open_files is not None:
            for f in self._open_files.values():
                f.close()
            self._open_files = None

        self._data_objects = None

    def __del__(self):
        self._close_open_files()

    def _look_for_files(self) -> Tuple[Dict[str, Dict], List[str], List[str]]:
        session_info_dict = {}
        session_ids = []
        unit_ids = []

        for i, selection_list in enumerate(self.include):
            selection = selection_list["selection"]
            config = selection_list.get("config", {})
            # parse selection
            if len(selection) == 0:
                raise ValueError(
                    f"Selection {i} is empty. Please at least specify a dandiset."
                )

            for subselection in selection:
                if subselection.get("dandiset", "") == "":
                    raise ValueError(f"Please specify a dandiset to include.")

                if self.task == 'drifting_gratings':
                    description_file = os.path.join(
                        self.root, subselection["dandiset"], "description_drifting_gratings.mpk"
                    )

                elif self.task == 'movie_decoding_one':
                    description_file = os.path.join(
                        self.root, subselection["dandiset"], "description_movie_decoding.mpk"
                    )

                elif self.task == 'movie_decoding_three':
                    description_file = os.path.join(
                        self.root, subselection["dandiset"], "description_movie_decoding_three.mpk"
                    )
                    
                elif self.task == 'static_gratings':
                    description_file = os.path.join(
                        self.root, subselection["dandiset"], "description_static_gratings.mpk"
                    )

                elif self.task == 'scene_decoding':
                    description_file = os.path.join(
                        self.root, subselection["dandiset"], "description_scene_decoding.mpk"
                    )
                elif self.task == 'stimulus':
                    description_file = os.path.join(
                        self.root, subselection["dandiset"], "description_food.mpk"
                    )
                elif self.task == 'stimulus_binary':
                    description_file = os.path.join(
                        self.root, subselection["dandiset"], "description_food_binary.mpk"
                    )

                try:
                    with open(description_file, "rb") as f:
                        description = msgpack.load(f, object_hook=decode_datetime)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Could not find description file {description_file}. This error "
                        "might be due to running an old pipeline that generates a "
                        "description.yaml file. Try running the appropriate snakemake "
                        "pipeline to generate the msgpack (mpk) file instead."
                    )
                # Get a list of all the potentially chunks in this dataset.
                sortsets = description["sortsets"]
                all_sortset_ids = [x["id"] for x in sortsets]
                all_sortset_subjects = set([x["subject"] for x in sortsets])

                # Perform selection. Right now, we are limiting ourselves to sortset,
                # subject and session, but we could make selection more flexible in the
                # future.
                sel_sortset = subselection.get("sortset", None)
                sel_sortsets = subselection.get("sortsets", None)
                sel_sortset_lte = subselection.get("sortset_lte", None)
                sel_subject = subselection.get("subject", None)
                sel_subjects = subselection.get("subjects", None)
                # exclude_sortsets allows you to exclude some sortsets from the selection.
                # example use: you want to train on the complete dandiset, but leave out
                # a few sortsets for evaluating transfer performance.
                sel_exclude_sortsets = subselection.get("exclude_sortsets", None)

                sel_session = subselection.get("session", None)
                sel_sessions = subselection.get("sessions", None)
                sel_exclude_sessions = subselection.get("exclude_sessions", None)
                sel_output = subselection.get("output", None)
                
                filtered = False
                if sel_sortset is not None:
                    assert (
                        sel_sortset in all_sortset_ids
                    ), f"Sortset {sel_sortset} not found in dandiset {subselection['dandiset']}"
                    sortsets = [
                        sortset for sortset in sortsets if sortset["id"] == sel_sortset
                    ]
                    filtered = True

                if sel_sortsets is not None:
                    assert (
                        not filtered
                    ), "Cannot specify sortset AND sortsets in selection"

                    # Check that all sortsets are in the dandiset.
                    for sortset in sel_sortsets:
                        assert (
                            sortset in all_sortset_ids
                        ), f"Sortset {sortset} not found in dandiset {subselection['dandiset']}"

                    sortsets = [
                        sortset for sortset in sortsets if sortset["id"] in sel_sortsets
                    ]
                    filtered = True

                if sel_sortset_lte is not None:
                    assert (
                        not filtered
                    ), "Cannot specify sortset_lte AND sortset(s) in selection"

                    sortsets = [
                        sortset
                        for sortset in sortsets
                        if sortset["id"] <= sel_sortset_lte
                    ]
                    filtered = True

                if sel_subject is not None:
                    assert (
                        not filtered
                    ), "Cannot specify subject AND sortset(s)/sortset_lte in selection"

                    assert (
                        sel_subject in all_sortset_subjects
                    ), f"Could not find subject {sel_subject} in dandiset {subselection['dandiset']}"

                    sortsets = [
                        sortset
                        for sortset in sortsets
                        if sortset["subject"] == sel_subject
                    ]
                    filtered = True

                if sel_subjects is not None:
                    assert (
                        not filtered
                    ), "Cannot specify subjects AND subject/sortset(s)/sortset_lte in selection"

                    # Make sure all subjects asked for are in the dandiset
                    sel_subjects = set(sel_subjects)
                    assert sel_subjects.issubset(all_sortset_subjects), (
                        f"Could not find subject(s) {sel_subjects - all_sortset_subjects} "
                        f" in dandiset {subselection['dandiset']}"
                    )

                    sortsets = [
                        sortset
                        for sortset in sortsets
                        if sortset["subject"] in sel_subjects
                    ]
                    filtered = True

                # Exclude sortsets if asked.
                if sel_exclude_sortsets is not None:
                    sortsets = [
                        sortset
                        for sortset in sortsets
                        if sortset["id"] not in sel_exclude_sortsets
                    ]

                # Note that this logic may result in adding too many slots but that's fine.
                unit_ids += [x for sortset in sortsets for x in sortset["units"]]
                # unit_ids are already fully qualified with prepended dandiset id.

                # Now we get the session-level information.
                sessions = sum([sortset["sessions"] for sortset in sortsets], [])

                filtered_on_session = False
                if sel_session is not None:
                    sessions = [
                        session for session in sessions if session["id"] == sel_session
                    ]
                    filtered_on_session = True

                if sel_sessions is not None:
                    assert (
                        not filtered_on_session
                    ), "Cannot specify session AND sessions in selection"
                    sessions = [
                        session for session in sessions if session["id"] in sel_sessions
                    ]

                if sel_exclude_sessions is not None:
                    sessions = [
                        session
                        for session in sessions
                        if session["id"] not in sel_exclude_sessions
                    ]

                assert (
                    len(sessions) > 0
                ), f"No sessions found for {i}'th selection included"

                # Similarly, select for certain outputs
                if sel_output is not None:
                    raise ValueError(
                        "Selecting dataset by 'output' is no longer possible."
                    )
                    # sessions = [
                    #     session
                    #     for session in sessions
                    #     if sel_output in session["fields"].keys()
                    # ]

                # Now we get the session-level information
                for session in sessions:
                    # iomap = {k: session[k] for k in ["fields", "task"]}
                    # # Check that the chunk has the requisite inputs.
                    # check = check_include(selection_list, iomap["fields"])
                    # if not check:
                    #     continue

                    session_id = subselection["dandiset"] + "/" + session["id"]

                    if session_id in session_info_dict:
                        raise ValueError(
                            f"Session {session_id} is already included in the dataset."
                            "Please verify that it is only selected once."
                        )

                    session_ids.append(session_id)

                    if self.pretrain or self.split == "all":
                        # pretrain or eval_only (all): use the full interval
                        train_list = session["splits"]["train"]
                        valid_list = session["splits"]["valid"]
                        test_list = session["splits"]["test"]

                        self.interval_list = sorted(train_list + valid_list + test_list, key=lambda x: x[0])
                        session_info_dict[session_id] = dict(
                            filename=(Path(self.root) / (session_id + ".h5")),
                            sampling_intervals=Interval.from_list(
                                self.interval_list
                            ),
                            config=config,
                            )

                    else:
                        intervals = session["splits"][self.split]  # self.split is one of: train, valid, test
                        session_info_dict[session_id] = dict(
                            filename=(Path(self.root) / (session_id + ".h5")),
                            sampling_intervals=Interval.from_list(intervals),
                            config=config,
                        )


        unit_ids = sorted(list(set(unit_ids)))
        session_ids = sorted(session_ids)
        return session_info_dict, session_ids, unit_ids

    def get(self, session_id: str, start: float, end: float):
        r"""This is the main method to extract a slice from a session. It returns a
        Data object that contains all data for session :obj:`session_id` between
        times :obj:`start` and :obj:`end`.

        Args:
            session_id: The session id of the slice. Note this is the fully qualified
                session-id: <dandiset>/<session_id>
            start: The start time of the slice.
            end: The end time of the slice.
        """
        data = copy.copy(self._data_objects[session_id])
        # TODO: add more tests to make sure that slice does not modify the original data object
        # note there should be no issues as long as the self._data_objects stay lazy
        sample = data.slice(start, end)
        session_info = self.session_info_dict[session_id]

        if self._check_for_data_leakage_flag:
            if self.pretrain or self.split == "all":
                sample._check_for_data_leakage('train')
                sample._check_for_data_leakage('valid')
                sample._check_for_data_leakage('test')
            else:
                sample._check_for_data_leakage(self.split)

        sample.session = session_id
        sample.config = session_info["config"]

        # Apply session-level z-score normalization
        if session_id in self._zscore_stats:
            mean, std = self._zscore_stats[session_id]
            sample.calcium_traces.df_over_f = (
                (sample.calcium_traces.df_over_f - mean) / std
            ).astype(np.float32)

        return sample

    def get_session_data(self, session_id: str):
        r"""Returns the data object corresponding to the session :obj:`session_id`.
        If the split is not "full", the data object is sliced to the allowed sampling
        intervals for the split, to avoid any data leakage. :obj:`RegularTimeSeries`
        objects are converted to :obj:`IrregularTimeSeries` objects, since they are
        most likely no longer contiguous.

        .. warning::
            This method might load the full data object in memory, avoid multiple calls
            to this method if possible.
        """
        data = copy.copy(self._data_objects[session_id])

        # get allowed sampling intervals
        if self.split != "full":
            # TODO(mehdi): in the future each object should hold its own sampling
            # intervals in the hdf5 file.
            sampling_intervals = self.get_sampling_intervals()[session_id]
            # print('sampling_intervals in get_session_data', sampling_intervals)


            sampling_intervals = Interval.from_list(sampling_intervals)
            data = data.select_by_interval(sampling_intervals)
            if self._check_for_data_leakage_flag:
                if self.pretrain or self.split == "all":
                    data._check_for_data_leakage('train')
                    data._check_for_data_leakage('valid')
                    data._check_for_data_leakage('test')
                else:
                    data._check_for_data_leakage(self.split)
        else:
            data = copy.deepcopy(data)
        return data

    def get_sampling_intervals(self):
        r"""Returns a dictionary of interval-list for each session.
        Each interval-list is a list of tuples (start, end) for each interval. This
        represents the intervals that can be sampled from each session.

        Note that these intervals will change depending on the split.
        """
        interval_dict = {}
        for session_id, session_info in self.session_info_dict.items():
            sampling_intervals_modifier_code = session_info["config"].get(
                "sampling_intervals_modifier", None
            )
            start = session_info["sampling_intervals"].start
            end = session_info["sampling_intervals"].end

            if sampling_intervals_modifier_code is None:
                if (end-start != 0).any():
                    interval_dict[session_id] = list(
                        zip(
                            session_info["sampling_intervals"].start,
                            session_info["sampling_intervals"].end,
                        )
                    )
                else:
                    continue
            else:
                local_vars = {
                    "data": copy.deepcopy(self._data_objects[session_id]),
                    "sampling_intervals": session_info["sampling_intervals"],
                    "split": self.split,
                    }
                #print('intervals', session_info["sampling_intervals"].end-session_info["sampling_intervals"].start)

                try:
                    exec(sampling_intervals_modifier_code, {}, local_vars)
                except NameError as e:
                    error_message = (
                        f"{e}. Variables that are passed to the sampling_intervals_modifier "
                        f"are: {list(local_vars.keys())}"
                    )
                    raise NameError(error_message) from e
                except Exception as e:
                    print('type of error', type(e))
                    print('name of error', str(e))
                    error_message = (
                        f"Error while executing sampling_intervals_modifier defined in "
                        f"the config file for session {session_id}: {e}"
                    )
                    import traceback
                    traceback.print_exc()
                    raise type(e)(error_message) from e

                sampling_intervals = local_vars.get("sampling_intervals")
                interval_dict[session_id] = list(
                    zip(sampling_intervals.start, sampling_intervals.end)
                )
        return interval_dict

    def disable_data_leakage_check(self):
        r"""Disables the data leakage check.

        .. warning::
            Only do this you are absolutely sure that there is no leakage between the
            current split and other splits (eg. the test split).
        """
        self._check_for_data_leakage_flag = False
        logging.warn(
            f"Data leakage check is disabled. Please be absolutely sure that there is "
            f"no leakage between {self.split} and other splits."
        )

    def __getitem__(self, index: DatasetIndex):
        ## index : DatasetIndex(session_id='allen_brain_observatory_calcium/599320182', start=338.40333, end=339.40333)
        ## self.valid_session_ids : ['allen_brain_observatory_calcium/599320182', ... ]
        try:
            # print(f"🟢 Loading index {index}")
            sample = self.get(index.session_id, index.start, index.end)
            if self.transform is not None: # apply transform
                sample = self.transform(sample) # tokenizer if applyed
        except Exception as e :
            print(f"❌ Exception in __getitem__({index}): {e}")
            import traceback
            traceback.print_exc()
            raise
        
        
        ## sample is dict!!
        if 'movie_decoding' in self.task:
            try:
                indices = sample['output_values'].obj['NATURAL_MOVIE_ONE'].squeeze().astype(int).tolist() # integer!
            except:
                #print('no indices', index.session_id)  # data too short to sample
                return None
            
            if sample['patches'].obj.shape[0] != sample['unit_index'].obj.shape[0]:
                patches = sample['patches']
                if hasattr(patches, 'obj'):
                    patches = patches.obj
                
                desired_length = sample['unit_index'].obj.shape[0]

                # pad if length is insufficient
                current_length = patches.shape[0]

                if current_length < desired_length:
                    pad_amount = desired_length - current_length
                    patches = F.pad(patches, (0, 0, 0, pad_amount))  # (left, right, top, bottom)
                else:
                    patches = patches[:desired_length, :]

                sample['patches'] = patches #PaddedObject(obj=patches)

            sample['movie_frames'] = torch.tensor(self.movie_frames[indices, :, :])


        elif self.task == 'scene_decoding':
            try:
                indices = sample['output_values'].obj['NATURAL_SCENES'].squeeze().astype(int).tolist()
            except:
                print('no indices', index.session_id)  # data too short to sample
                return None

            if sample['patches'].obj.shape[0] != sample['unit_index'].obj.shape[0]:
                patches = sample['patches']
                if hasattr(patches, 'obj'):
                    patches = patches.obj

                desired_length = sample['unit_index'].obj.shape[0]

                # pad if length is insufficient
                current_length = patches.shape[0]

                if current_length < desired_length:
                    pad_amount = desired_length - current_length
                    patches = F.pad(patches, (0, 0, 0, pad_amount))  # (left, right, top, bottom)
                else:
                    patches = patches[:desired_length, :]

                sample['patches'] = patches #PaddedObject(obj=patches)

            sample['movie_frames'] = torch.tensor(self.movie_frames[indices, :, :])

        # print('keys of sample is:', sample.keys())
        # 'unit_index', 'timestamps', 'patches', 'token_type', 'input_mask', 'latent_index', 'latent_timestamps',
        # 'unit_spatial_emb', 'unit_feats', -> cannot find in fine-tuning
        # 'unit_cre_line', 'session_index',
        # 'output_timestamps', 'output_decoder_index', 'output_values', 'output_weights'
         

        return sample

    def __len__(self):
        return len(self.valid_session_ids)
        #raise NotImplementedError("Length of dataset is not defined")

    def __iter__(self):
        raise NotImplementedError("Iteration over dataset is not defined")


def decode_datetime(obj):
    """msgpack doesn't support datetime, so we need to encode it as a string."""
    if "__datetime__" in obj:
        return datetime.datetime.fromisoformat(obj["as_str"])
    return obj
