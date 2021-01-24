function y = bads_ll_greedy(x,sub)

    y = py.bads_ibs_greedy.ibs_interface(x(1), x(2), x(3), x(4), x(5), x(6), sub);
    disp(y);

end