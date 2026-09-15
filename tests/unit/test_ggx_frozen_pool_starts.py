import pytest

from tools.audit_ggx_frozen_pool_starts import parse_args, select_first, validate_pool


def fixtures():
    rows = [dict(rank=i, confidence=1-i/100, official_scene_clear=True,
                 local_state='geometry_eligible' if i in (20, 26, 31) else 'local_rejected')
            for i in range(100)]
    return dict(feasibility_stages=rows, eligible_ranks=[20, 26, 31]), dict(candidates=rows)


def test_original_rank_first_pass_and_complete_ledger():
    pool, official = fixtures()
    rows = validate_pool(pool, official)
    called = []
    def attempt(rank):
        called.append(rank)
        return dict(status='nominal_route_pass' if rank == 26 else 'nominal_route_rejected')
    selected, ledger = select_first(rows, attempt)
    assert selected == 26 and called == [20, 26]
    assert len(ledger) == 100
    assert ledger[31]['status'] == 'not_tested_after_first_pass'
    assert ledger[0]['status'] == 'saved_screen_rejection'


def test_exhaustion_is_not_physical_failure():
    pool, _ = fixtures()
    selected, ledger = select_first(pool['feasibility_stages'], lambda _: dict(status='nominal_route_rejected'))
    assert selected is None
    assert sum(r['status'] == 'nominal_route_rejected' for r in ledger) == 3


def test_rejects_truncation_and_reranking():
    pool, official = fixtures()
    pool['feasibility_stages'] = pool['feasibility_stages'][::-1]
    with pytest.raises(ValueError, match='original top-100'):
        validate_pool(pool, official)


def test_unexpected_errors_are_not_candidate_rejections():
    pool, _ = fixtures()
    def broken(_):
        raise ValueError('corrupt input')
    with pytest.raises(ValueError, match='corrupt input'):
        select_first(pool['feasibility_stages'], broken)


def test_no_execute_switch():
    with pytest.raises(SystemExit):
        parse_args(['--execute'])
